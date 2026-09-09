import collections

import torch
import torch.distributed as dist
from implementation.startup import validate_router_aux_loss_coef


def backward_with_router_loss(output, grad, auxiliary, coefficient, split_num):
    # One traversal keeps the FSDP weight-lifetime hooks correct.
    if auxiliary is None:
        output.backward(grad)
    else:
        torch.autograd.backward((output, auxiliary), (grad, auxiliary.new_tensor(coefficient / split_num)))


def router_loss_metrics(task_loss, auxiliary, coefficient, pp_group, dp_group):
    metrics = torch.stack((torch.as_tensor(task_loss, device=auxiliary.device).float(), auxiliary))
    if pp_group is not None and pp_group.size() > 1:
        dist.all_reduce(metrics, group=pp_group)
    if dp_group is not None and dp_group.size() > 1:
        dist.all_reduce(metrics, group=dp_group)
        metrics /= dp_group.size()
    return {"task_loss": metrics[0], "router_aux_loss": metrics[1],
            "total_loss": metrics[0] + coefficient * metrics[1]}


def pipelined_train_overlap(model, optimizer, x, x_label, x_spec: list[int], split_num, pipeline_stage_this, loss_fn, dtype=None, device=None, pp_group: dist.ProcessGroup = None, dp_group: dist.ProcessGroup = None, router_aux_loss_coef=0.0, return_metrics=False):
    validate_router_aux_loss_coef(router_aux_loss_coef)
    if split_num <= 0:
        raise ValueError("split_num must be positive")
    if x_spec[0] % split_num != 0:
        raise ValueError(f"batch size {x_spec[0]} must be divisible by split_num {split_num}")
    pipeline_stage_total = 1 if pp_group is None else pp_group.size()
    if not 0 <= pipeline_stage_this < pipeline_stage_total:
        raise ValueError("pipeline_stage_this must identify a stage in pp_group")
    first_stage = pipeline_stage_this == 0
    last_stage = pipeline_stage_this == pipeline_stage_total - 1
    ranks = dist.get_process_group_ranks(pp_group) if pp_group is not None else []
    split_size = x_spec[0] // split_num
    microbatch_spec = [split_size, *x_spec[1:]]
    tensor_chunks = torch.split(x, split_size) if first_stage else None
    label_chunks = torch.split(x_label, split_size) if last_stage else None
    inputs = collections.deque()
    outputs = collections.deque()
    auxiliary_total = torch.zeros((), device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    model.clear_grad_accumulators()
    loss_total = 0.0

    def exchange(send_tensor=None, receive=False, downstream=True):
        peer = ranks[pipeline_stage_this + (1 if downstream else -1)]
        received = torch.empty(microbatch_spec, dtype=dtype, device=device) if receive else None
        ops = []
        send_buffer = None
        if send_tensor is not None:
            send_buffer = send_tensor.detach().contiguous()
            ops.append(dist.P2POp(dist.isend, send_buffer, peer, group=pp_group))
        if receive:
            ops.append(dist.P2POp(dist.irecv, received, peer, group=pp_group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return received

    def forward_microbatch(x_in):
        if not first_stage:
            x_in.requires_grad_(True)
        inputs.append(x_in)
        x_out, auxiliary = model.forward(x_in, return_aux_loss=True) if router_aux_loss_coef else (model.forward(x_in), None)
        outputs.append((x_out, auxiliary))
        return x_out

    def backward_microbatch(grad, microbatch_index):
        nonlocal loss_total
        x_in = inputs.popleft()
        x_out, auxiliary = outputs.popleft()
        if auxiliary is not None:
            auxiliary_total.add_(auxiliary.detach() / split_num)
        if last_stage:
            loss = loss_fn(x_out, label_chunks[microbatch_index])
            loss_total += loss.detach() / split_num
            backward_with_router_loss(loss / split_num, None, auxiliary, router_aux_loss_coef, split_num)
        else:
            backward_with_router_loss(x_out, grad, auxiliary, router_aux_loss_coef, split_num)
        model.accumulate_full_gradients()
        return None if first_stage else x_in.grad

    warmup_microbatches = min(pipeline_stage_total - pipeline_stage_this - 1, split_num)
    remaining_microbatches = split_num - warmup_microbatches


    for i in range(warmup_microbatches):
        x_in = tensor_chunks[i] if first_stage else exchange(receive=True, downstream=False)
        x_out = forward_microbatch(x_in)
        exchange(send_tensor=x_out)

    if remaining_microbatches:
        x_in = tensor_chunks[warmup_microbatches] if first_stage else exchange(receive=True, downstream=False)
    for i in range(remaining_microbatches):
        x_out = forward_microbatch(x_in)
        grad = None if last_stage else exchange(send_tensor=x_out, receive=True)
        input_grad = backward_microbatch(grad, i)
        has_next_forward = i + 1 < remaining_microbatches
        if not first_stage:
            x_in = exchange(send_tensor=input_grad, receive=has_next_forward, downstream=False)
        elif has_next_forward:
            x_in = tensor_chunks[warmup_microbatches + i + 1]

    for i in range(remaining_microbatches, split_num):
        grad = exchange(receive=True)
        input_grad = backward_microbatch(grad, i)
        if not first_stage:
            exchange(send_tensor=input_grad, downstream=False)

    model.reduce_accumulated_gradients()
    optimizer.step()
    if return_metrics:
        return router_loss_metrics(loss_total, auxiliary_total, router_aux_loss_coef, pp_group, dp_group)
    return loss_total

