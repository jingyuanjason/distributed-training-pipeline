import json
import random
import torch
from torch import nn
import yaml
import wandb

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.train_utils import compute_rollout_rewards, grpo_train_step
from cs336_alignment.vllm_utils import generate_completions, init_weight_sync, kill_existing_vllm_server, start_server, sync_policy_weights, wait_for_server


def start_vllm_server(model_id, vllm_host, gpu_num=0, port=8088):
    vllm_base_url = f"http://{vllm_host}:{port}"
    kill_existing_vllm_server(8088)
    process = start_server(model_id, vllm_host, port, gpu_num, seed=0, load_format="auto", logging_level="ERROR")
    print("Waiting for vllm server to be ready")
    wait_for_server(vllm_base_url, process, 300)
    print("Vllm server is ready!")

def load_config(train_config_path = "cs336_alignment/run_config.yaml"):
    config = {}
    with open(train_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def generate_result(prompts: list[str]=[], stopper=True, vllm_base_url="http://localhost:8088", model_id="allenai/OLMo-2-0425-1B", group_size=1, temperature=1.0):
    assert len(prompts) != 0
    sampling_params = {}
    sampling_params["temperature"] = temperature
    sampling_params["max_tokens"] = 512
    sampling_params["n"] = group_size
    sampling_params["seed"] = 0
    if stopper:
        sampling_params['stop'] = ["</answer>"]
        sampling_params['include_stop_str_in_output'] = True
    return generate_completions(vllm_base_url, model_id, prompts, sampling_params)

def load_prompts(prompt_path):

    prompt = ""
    with open(prompt_path, "r") as f:
        prompts = "\n".join(f.readlines())
    return prompts


def load_dataset(dataset_path:str):
    dataset_raw = []
    raw_json = []
    with open(dataset_path, "r") as f:
        raw_json = f.readlines()
    for line in raw_json:
        dataset_raw.append(json.loads(line))
    return dataset_raw

def compose_input(prompt, question):
    return prompt.replace("{question}", question)

def get_reward_func(reward_func_name):
    if reward_func_name == "r1_zero_reward_fn":
        return r1_zero_reward_fn
    
def get_optimizer(config, model: nn.Module):
    optimizer_config = config["optimizer"]
    optmizier_type = optimizer_config["type"]
    optimizer_params = optimizer_config["params"]
    if optmizier_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    return optimizer


def get_question_and_answer_sampler(dataset, prompt, shuffle=True, init_index=0, infinite=True):
    data_indices = [i for i in range(len(dataset))]
    if shuffle:
        random.shuffle(data_indices)
    current_index = init_index
    
        
    def get_sample(batch_size=256, group_size=8, start_idx=-1, vllm_base_url="http://localhost:8088", temperature=1.0):
        print(f"sampling {batch_size} questions, with group size {group_size}")
        nonlocal current_index, infinite
        if start_idx != -1:
            current_index = start_idx
        data_indices_this = [data_indices[idx % len(dataset)] for idx in range(current_index, current_index + batch_size)]
        current_index += batch_size
        train_data_batch = [dataset[i] for i in data_indices_this]
        questions_w_prompt = [compose_input(prompt, train_data["question"]) for train_data in train_data_batch]
        
        answers = [train_data["answer"].split("####")[1].strip() for train_data in train_data_batch for _ in range(group_size)]
        generated_answers = generate_result(questions_w_prompt, vllm_base_url=vllm_base_url, group_size=group_size, temperature=temperature)
        generated_answers = [completion.text for completion in generated_answers]
        questions_w_prompt = [question for question in questions_w_prompt for _ in range(group_size)]
        return questions_w_prompt, answers, generated_answers
    return get_sample
    
def get_samplers(config):
    eval_dataset_path = config["data_source"]["eval_dataset_path"]
    train_dataset_path = config["data_source"]["train_dataset_path"]
    prompt_path = config["data_source"]["prompt_path"]
    prompt = load_prompts(prompt_path)
    train_dataset = load_dataset(train_dataset_path)
    eval_dataset = load_dataset(eval_dataset_path)
    return get_question_and_answer_sampler(train_dataset, prompt), get_question_and_answer_sampler(eval_dataset, prompt, False, 0)

def train(run):
    config = load_config()
    
    batch_size = config["training_progress"]["batch_size"]
    gradient_accumulation_steps = config["training_progress"]["gradient_accumulation_steps"]
    target_iteration = config["training_progress"]["target_iteration"]
    validation_interval = config["training_progress"]["validation_interval"]
    rollout_steps = config["training_progress"]["rollout_steps"]

    rl_config = config["rl_param"]
    reward_func_name = rl_config["sampling"]["reward_func_name"]
    group_size = rl_config["sampling"]["group_size"]
    temperature = rl_config["sampling"]["temperature"]
    baseline = rl_config["loss_formation"]["baseline"]
    advantage_eps = rl_config["loss_formation"]["advantage_eps"]
    advantage_normalizer = rl_config["loss_formation"]["advantage_normalizer"]
    importance_reweighting_method = rl_config["loss_formation"]["importance_reweighting_method"]
    cliprange = rl_config["loss_formation"]["cliprange"]
    loss_normalization = rl_config["loss_formation"]["loss_normalization"]
    normalization_constant = rl_config["loss_formation"]["normalization_constant"]

    max_grad_norm = config["optimizer"]["gradient_clip"]["max_grad_norm"]

    model_id = config["model"]["id"]
    model_device = config["model"]["device"]

    start_vllm = config["start_vllm"]
    print_only = config["print_only"]
    vllm_host =  config["vllm_server"]["host"]
    vllm_port =  config["vllm_server"]["port"]
    vllm_gpu = config["vllm_server"]["gpu"]
    vllm_base_url = f"http://{vllm_host}:{vllm_port}"

    if not print_only:
        if start_vllm:
            start_vllm_server(model_id, vllm_host=vllm_host, port=vllm_port, gpu_num=vllm_gpu)
        model, tokenizer = get_model_and_tokenizer(model_id, model_device)
        optimizer = get_optimizer(config, model)
        weight_sync_group = init_weight_sync(vllm_base_url, model_device)
        sync_policy_weights(model, vllm_base_url, weight_sync_group)
    print("Start training", flush=True)
    reward_func = get_reward_func(reward_func_name)

    train_data_sampler, eval_data_sampler = get_samplers(config)
    for iter_this in range(1, target_iteration+1): # fix to right start
        loss_acc = 0.0
        reward_acc = 0.0
        for rollout_step in range(rollout_steps):
            questions_w_prompt, answers, generated_answers = train_data_sampler(batch_size=batch_size, group_size=group_size, vllm_base_url=vllm_base_url, temperature=temperature)
            reward = compute_rollout_rewards(reward_func, generated_answers, answers)[0].mean()

            old_log_probs = None
            loss_this, _ = grpo_train_step(model=model, 
                                tokenizer=tokenizer, 
                                optimizer=optimizer, 
                                gradient_accumulation_steps=gradient_accumulation_steps, 
                                max_grad_norm=max_grad_norm, 
                                reward_fn=reward_func,
                                repeated_prompts=questions_w_prompt,
                                rollout_responses=generated_answers,
                                repeated_ground_truths=answers,
                                group_size=group_size,
                                baseline = baseline,
                                advantage_eps = advantage_eps,
                                advantage_normalizer = advantage_normalizer,
                                importance_reweighting_method = importance_reweighting_method,
                                old_log_probs = old_log_probs,
                                cliprange = cliprange,
                                loss_normalization = loss_normalization,
                                normalization_constant = normalization_constant
                                )
            loss_acc += loss_this
            reward_acc += reward
        run.log({
            "train/loss": loss_acc/rollout_steps,
            "step": iter_this,
            "reward_train": reward_acc/rollout_steps,
        })
        sync_policy_weights(model, vllm_base_url, weight_sync_group)
        if iter_this % validation_interval == 0:

            questions_w_prompt, answers, generated_answers = eval_data_sampler(batch_size=1024, group_size=1, start_idx=0, vllm_base_url=vllm_base_url, temperature=0.0)
            reward = compute_rollout_rewards(reward_func, generated_answers, answers)
            run.log({
                "step": iter_this,
                "eval_reward": reward[0].mean(),
            })
            print(f"average reward at iteration: {iter_this} is {reward[0].mean()}", flush=True)


if __name__ == "__main__":
    with wandb.init(
                    project="rl_train",
                    config={
                        "learning_rate": 1e-3,
                        "batch_size": 64,
                        "epochs": 10,
                    },
                ) as run:
        train(run)
    # training loop
        # sample questions
        # generate outputs
        # run training steps
        # sync weight