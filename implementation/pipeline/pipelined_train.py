import collections

import torch
import torch.distributed as dist
from implementation.pipeline.pipelined_train_overlap import backward_with_router_loss, router_loss_metrics
from implementation.startup import validate_router_aux_loss_coef


def pipelined_train(model, optimizer, x, x_label, x_spec: list[int], split_num, pipeline_stage_this, loss_fn, dtype=None, device=None, pp_group: dist.ProcessGroup = None, dp_group: dist.ProcessGroup = None, router_aux_loss_coef=0.0, return_metrics=False):
    validate_router_aux_loss_coef(router_aux_loss_coef)
    if pp_group is None:
        pipeline_stage_total = 1
    else:
        pipeline_stage_total = pp_group.size()
    split_size = x_spec[0]//split_num
    x_spec[0] = x_spec[0]//split_num
    if pipeline_stage_this == 0:
        tensor_chunks = torch.split(x, split_size)
    else:
        tensor_chunks = None

    if pipeline_stage_this == pipeline_stage_total - 1:
        label_chunks = torch.split(x_label, split_size)
    else:
        label_chunks = None
    
    ranks = dist.get_process_group_ranks(pp_group)
    loss = None
    inputs = collections.deque()
    outputs = collections.deque()
    auxiliary_total = torch.zeros((), device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    model.clear_grad_accumulators()

    warmup_microbatches = min(
        pipeline_stage_total - pipeline_stage_this - 1,
        split_num,
    )

    loss_total = 0.0

    activation_send_work = None
    activation_recv_work = None
    grad_send_work = None
    grad_recv_work = None
    for i in range(warmup_microbatches):
        if pipeline_stage_this == 0:
            x_in = tensor_chunks[i]
        else:
            x_in = torch.empty(x_spec, dtype=dtype, device=device)
            activation_recv_work = dist.irecv(x_in, src=ranks[pipeline_stage_this-1])
            activation_recv_work.wait()
            x_in.requires_grad_(True)

        inputs.append(x_in)
        x_out, auxiliary = model.forward(x_in, return_aux_loss=True) if router_aux_loss_coef else (model.forward(x_in), None)
        outputs.append((x_out, auxiliary))

        activation_send_work = dist.isend(x_out.detach(), ranks[pipeline_stage_this+1])
        activation_send_work.wait()

    if pipeline_stage_this != pipeline_stage_total - 1:
        grad_input = torch.empty(x_spec, dtype=dtype, device=device)

    for i in range(warmup_microbatches, split_num):

        if pipeline_stage_this != 0:
            x_in = torch.empty(x_spec, dtype=dtype, device=device)
            activation_recv_work = dist.irecv(x_in, src=ranks[pipeline_stage_this-1])

        if pipeline_stage_this != pipeline_stage_total - 1:
            grad_recv_work = dist.irecv(grad_input, src=ranks[pipeline_stage_this+1])

        if pipeline_stage_this == 0:
            x_in = tensor_chunks[i]
        else:
            activation_recv_work.wait()
            x_in.requires_grad_(True)
        inputs.append(x_in)
        x_out, auxiliary = model.forward(x_in, return_aux_loss=True) if router_aux_loss_coef else (model.forward(x_in), None)
        outputs.append((x_out, auxiliary))
        if pipeline_stage_this != pipeline_stage_total - 1:
            activation_send_work = dist.isend(x_out.detach(), ranks[pipeline_stage_this+1])
        else:
            loss = loss_fn(x_out, label_chunks[i])
            loss_total += loss.detach()/split_num

        x_in_grad: torch.Tensor = inputs.popleft()
        x_out_grad, auxiliary = outputs.popleft()
        if auxiliary is not None:
            auxiliary_total.add_(auxiliary.detach() / split_num)
        if pipeline_stage_this == pipeline_stage_total - 1:
            grad_input = loss
            backward_with_router_loss(loss / split_num, None, auxiliary, router_aux_loss_coef, split_num)
        else:
            grad_recv_work.wait()
            backward_with_router_loss(x_out_grad, grad_input, auxiliary, router_aux_loss_coef, split_num)
        model.accumulate_full_gradients()

        if pipeline_stage_this != 0:
            grad_send_work = dist.isend(x_in_grad.grad, ranks[pipeline_stage_this-1])
            grad_send_work.wait()
        if pipeline_stage_this != pipeline_stage_total - 1:
            activation_send_work.wait()

    for i in range(split_num - warmup_microbatches, split_num):
        x_in_grad: torch.Tensor = inputs.popleft()
        x_out_grad, auxiliary = outputs.popleft()
        if auxiliary is not None:
            auxiliary_total.add_(auxiliary.detach() / split_num)

        grad_recv_work = dist.irecv(grad_input, src=ranks[pipeline_stage_this+1])
        grad_recv_work.wait()
        backward_with_router_loss(x_out_grad, grad_input, auxiliary, router_aux_loss_coef, split_num)
        model.accumulate_full_gradients()
        if pipeline_stage_this != 0:
            grad_send_work = dist.isend(x_in_grad.grad, ranks[pipeline_stage_this-1])
            grad_send_work.wait()

    model.reduce_accumulated_gradients()
    optimizer.step()
    if return_metrics:
        return router_loss_metrics(loss_total, auxiliary_total, router_aux_loss_coef, pp_group, dp_group)
    return loss_total

