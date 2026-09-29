from typing import Callable, Literal

from einops import rearrange
from functorch import einops
import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer


def tokenize_prompt_and_output(prompt_strs: list[str], 
                               output_strs: list[str], 
                               tokenizer: PreTrainedTokenizer,
                               ) -> dict[str, torch.Tensor]:
    prompt_tokens = [tokenizer.encode(prompt_str) for prompt_str in prompt_strs]
    output_tokens = [tokenizer.encode(output_str) for output_str in output_strs]
    all_strs = [prompt_token + output_token for prompt_token, output_token in zip(prompt_tokens, output_tokens)]
    masks = [[0] * len(prompt_token) + [1] * len(output_token) for prompt_token, output_token in zip(prompt_tokens, output_tokens)]
    max_len = max(len(s) for s in all_strs)
    list(map(lambda s:s.extend([0] * (max_len - len(s))), all_strs))
    list(map(lambda s:s.extend([0] * (max_len - len(s))), masks))
    data_tensor = torch.Tensor(all_strs).to(torch.int)
    mask_tensor = torch.Tensor(masks).to(torch.int)
    input_ids = data_tensor[...,:-1]
    labels = data_tensor[...,1:]
    response_mask = mask_tensor[...,1:]
    return {"input_ids":input_ids, "labels":labels, "response_mask":response_mask}

def get_response_log_probs(model: PreTrainedModel,
                           input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           return_token_entropy: bool = False,
                        ) -> dict[str, torch.Tensor]:
    y_pred = model(input_ids).logits # batch, seqlen, num_token
    
    y_pred = y_pred - y_pred.max(dim = -1, keepdim=True)[0]
    y_pred_exp = torch.exp(y_pred)
    exp_sum = torch.sum(y_pred_exp, dim=-1, keepdim=True)
    logexpsum = torch.log(exp_sum)
    
    logits_label = torch.gather(y_pred, dim=2, index=labels.unsqueeze(dim=-1)) # logits
    log_probs = logits_label - logexpsum
    results = {"log_probs": log_probs.squeeze(dim=-1)}
    if return_token_entropy:
        y_pred_exp = torch.exp(y_pred)
        token_prob = y_pred_exp/exp_sum
        token_entropy = -(token_prob * torch.log(token_prob)).sum(dim=-1)
        results["token_entropy"] = token_entropy
    return results

def compute_rollout_rewards(
                            reward_fn: Callable[[str, str], dict[str, float]],
                            rollout_responses: list[str],
                            repeated_ground_truths: list[str],
                            ) -> tuple[torch.Tensor, dict[str, float]]:
    
    rewards = [reward_fn(rollout_response, repeated_ground_truth) for rollout_response, repeated_ground_truth in zip(rollout_responses, repeated_ground_truths)]
    rewards_total = torch.Tensor([reward["reward"] for reward in rewards])
    rewards_format = torch.Tensor([reward["format_reward"] for reward in rewards])
    metadata = {}
    metadata["mean"] = rewards_total.mean()
    metadata["format_reward"] = rewards_format

    return rewards_total, metadata

def compute_group_normalized_rewards(
                                    raw_rewards: torch.Tensor,
                                    group_size: int,
                                    baseline: Literal["mean", "none"] = "mean",
                                    advantage_eps: float = 1e-6,
                                    advantage_normalizer: Literal["std", "none", "mean"] = "std",
                                    ):
    raw_rewards = rearrange(raw_rewards, "(b g) -> b g ", g=group_size)
    baseline = raw_rewards.mean(dim=-1, keepdim=True) if baseline == "mean" else 0
    if advantage_normalizer == "std":
        normalizer = raw_rewards.std(dim=-1, keepdim=True) + advantage_eps if advantage_normalizer == "std" else 1
    elif advantage_normalizer == "mean":
        normalizer = raw_rewards.mean(dim=-1, keepdim=True)
    else:
        normalizer = 1
    reward_normalized = (raw_rewards - baseline)/normalizer
    metadata = {}
    metadata["std"] = normalizer
    metadata["mean"] = baseline
    reward_normalized = rearrange(reward_normalized, "b g->(b g)")
    return reward_normalized, metadata

def compute_policy_gradient_loss(
                                raw_rewards_or_advantages: torch.Tensor,
                                policy_log_probs: torch.Tensor,
                                importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
                                old_log_probs: torch.Tensor | None = None,
                                cliprange: float | None = None,
                                response_mask: torch.Tensor | None = None,
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if importance_reweighting_method == "none":
        loss = (raw_rewards_or_advantages * policy_log_probs)
    elif importance_reweighting_method == "noclip":
        w = torch.exp(policy_log_probs - old_log_probs)
        loss = raw_rewards_or_advantages * w
    elif importance_reweighting_method == "grpo":
        w = torch.exp(policy_log_probs - old_log_probs)
        clipped = torch.minimum(w * raw_rewards_or_advantages, torch.clamp(w, 1 - cliprange, 1 + cliprange) * raw_rewards_or_advantages)
        loss = clipped
    else:
        L = response_mask.sum(dim=-1, keepdim=True)
        print(policy_log_probs.shape, old_log_probs.shape, response_mask.shape)
        w = torch.exp(((policy_log_probs - old_log_probs)*response_mask).sum(dim=-1, keepdim=True)/L) * torch.ones_like(policy_log_probs)
        
        clipped = torch.minimum(w * raw_rewards_or_advantages, torch.clamp(w, 1 - cliprange, 1 + cliprange) * raw_rewards_or_advantages)
        loss = clipped
        
    metadata = {}
    metadata["mean"] = loss.mean()
    metadata["std"] = loss.std()
    return -loss, metadata

def aggregate_loss_across_microbatch(
                                    per_token_policy_gradient_loss: torch.Tensor,
                                    mask: torch.Tensor,
                                    loss_normalization: Literal["sequence", "constant"] = "sequence",
                                    normalization_constant: int | None = None,
                                    ) -> torch.Tensor:
    
    loss_normalized = (per_token_policy_gradient_loss * mask).sum(dim=-1)
    if loss_normalization == "constant":
        assert normalization_constant != None
        loss_normalized = (loss_normalized / normalization_constant).sum()
    else:
        loss_normalized = (loss_normalized / mask.sum(dim=-1)).mean()
    return loss_normalized

def grpo_train_step(
                    model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    optimizer: Optimizer,
                    gradient_accumulation_steps: int,
                    max_grad_norm: float | None,
                    reward_fn: Callable[[str, str], dict[str, float]],
                    repeated_prompts: list[str],
                    rollout_responses: list[str],
                    repeated_ground_truths: list[str],
                    group_size: int,
                    # Reward normalization
                    baseline: Literal["mean", "none"] = "mean",
                    advantage_eps: float = 1e-6,
                    advantage_normalizer: Literal["std", "none", "mean"] = "std",
                    # Importance reweighting and clipping
                    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
                    old_log_probs: torch.Tensor | None = None,
                    cliprange: float | None = None,
                    # Loss normalization
                    loss_normalization: Literal["sequence", "constant"] = "sequence",
                    normalization_constant: int | None = None,
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    device = model.device
    token_dict = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer=tokenizer)
    input_token_all = token_dict["input_ids"].to(device)
    labels_all = token_dict["labels"].to(device)
    masks_all = token_dict["response_mask"].to(device)
    micro_batch_size = input_token_all.shape[0]//gradient_accumulation_steps
    loss_total = 0
    optimizer.zero_grad()
    for i in range(gradient_accumulation_steps):
        input_token_microbatch = input_token_all[i*micro_batch_size:i*micro_batch_size + micro_batch_size]
        labels_microbatch = labels_all[i*micro_batch_size:i*micro_batch_size + micro_batch_size]
        masks_microbatch = masks_all[i*micro_batch_size:i*micro_batch_size + micro_batch_size]
        rollout_responses_microbatch = rollout_responses[i*micro_batch_size:i*micro_batch_size + micro_batch_size]
        old_log_probs_microbatch = 0
        if old_log_probs is not None:
            old_log_probs_microbatch = old_log_probs[i*micro_batch_size:i*micro_batch_size + micro_batch_size]
        repeated_ground_truths_microbatch = repeated_ground_truths[i*micro_batch_size:i*micro_batch_size + micro_batch_size]

        log_prob_dict = get_response_log_probs(model, input_token_microbatch, labels_microbatch, return_token_entropy=True)
        log_prob_microbatch = log_prob_dict["log_probs"]

        reward, _ = compute_rollout_rewards(reward_fn, rollout_responses=rollout_responses_microbatch, repeated_ground_truths=repeated_ground_truths_microbatch)
        reward_normalized, _ = compute_group_normalized_rewards(reward, group_size=group_size, baseline=baseline, advantage_eps=advantage_eps, advantage_normalizer=advantage_normalizer)
        reward_normalized = reward_normalized.to(device)

        loss_token, _ = compute_policy_gradient_loss(reward_normalized.unsqueeze(dim=-1), log_prob_microbatch, importance_reweighting_method=importance_reweighting_method, old_log_probs=old_log_probs_microbatch, cliprange=cliprange, response_mask=masks_microbatch)
        if loss_normalization == "sequence":
            loss_normalized = aggregate_loss_across_microbatch(loss_token, masks_microbatch, loss_normalization=loss_normalization, normalization_constant=normalization_constant)/gradient_accumulation_steps
        elif loss_normalization == "constant":
            loss_normalized = aggregate_loss_across_microbatch(loss_token, masks_microbatch, loss_normalization=loss_normalization, normalization_constant=normalization_constant)
        loss_total += loss_normalized.detach()
        loss_normalized.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    metadata = {}
    return loss_total, metadata

