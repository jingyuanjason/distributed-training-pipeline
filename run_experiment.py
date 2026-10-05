"""Profile a custom workload using the run_config.yaml experiment section.

Edit module_wrapper below to supply your module and inputs. The only CLI option
is --config-path, defaulting to configs/run_config.yaml alongside this file.

The Python API profile_callable(lambda: module(*args, **kwargs)) accepts arbitrary
inputs through a closure. It preserves the caller's autograd and training modes.
Work must run on the selected device's current stream, or join that stream before
returning. CUDA event times measure elapsed device-stream time, not summed kernel
durations. Memory statistics cover only PyTorch's allocator, not all GPU memory.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml

ITERATIONS = 100
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "run_config.yaml"


def cuda_device(device="cuda:0"):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Profiling requires an NVIDIA CUDA device, not CPU.")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU and CUDA-enabled PyTorch are required.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is not available.")
    return device


def profile_callable(step, *, device="cuda:0", warmup=10, iterations=ITERATIONS):
    """Warm up, then profile calls; discard each output to avoid accumulation.

    Existing allocations (including model/inputs and warm-up caches) form the
    baseline. No cache is emptied between iterations, reflecting steady-state use.
    This function resets the selected device's allocator peak statistics.
    """
    if not callable(step):
        raise TypeError("step must be callable")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be nonnegative integer")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    device = cuda_device(device)
    with torch.cuda.device(device):
        for _ in range(warmup):
            output = step()
            del output
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Initialize events before measurement.
        start.record()
        end.record()
        end.synchronize()
        baseline = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(iterations):
            start.record()
            output = step()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            del output
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        mib = 1024 ** 2
        return {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "iterations": iterations,
            "warmup_iterations": warmup,
            "time_ms": {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stddev": statistics.pstdev(times),
                "min": min(times),
                "max": max(times),
                "total": sum(times),
            },
            "memory_mib": {
                "baseline_allocated": baseline / mib,
                "peak_allocated": peak / mib,
                "peak_above_baseline": (peak - baseline) / mib,
                "baseline_reserved": baseline_reserved / mib,
                "peak_reserved": torch.cuda.max_memory_reserved(device) / mib,
            },
        }


def load_experiment_config(path):
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("Configuration must contain an experiment section")  # noqa: TRY004
    experiment = config["experiment"]
    for key, minimum in (("warmup", 0), ("iterations", 1)):
        value = experiment.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"experiment.{key} must be an integer >= {minimum}")
    if experiment.get("dtype") not in ("float32", "float16", "bfloat16"):
        raise ValueError("experiment.dtype must be float32, float16, or bfloat16")
    if not isinstance(experiment.get("device"), str):
        raise ValueError("experiment.device must be a CUDA device string")  # noqa: TRY004
    if experiment.get("mode") not in ("training", "evaluation"):
        raise ValueError("experiment.mode must be training or evaluation")
    shapes = experiment.get("input_shapes")
    if not isinstance(shapes, list) or any(
        not isinstance(shape, list) or not shape
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        for shape in shapes
    ):
        raise ValueError("experiment.input_shapes must be a list of positive integer shapes")
    return config


def make_module_step(module, inputs, mode, loss_fn=None):
    """Build a forward/backward or no-grad forward step for positional inputs.

    Training includes loss reduction and gradient cleanup, but no optimizer step.
    The default loss is the mean of a single floating output tensor; supply
    loss_fn(output) for structured outputs or a task-specific scalar loss.
    Floating leaf inputs require gradients in training to include input-gradient
    computation. Rebuild the step/inputs when changing modes.
    """
    if mode not in ("training", "evaluation"):
        raise ValueError("mode must be training or evaluation")
    training = mode == "training"
    module.train(training)
    for tensor in inputs:
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.is_leaf:
            tensor.requires_grad_(training)

    def clear_gradients():
        module.zero_grad(set_to_none=True)
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor) and tensor.is_leaf:
                tensor.grad = None

    clear_gradients()

    def step():
        if not training:
            with torch.no_grad():
                return module(*inputs)
        with torch.enable_grad():
            try:
                output = module(*inputs)
                loss = loss_fn(output) if loss_fn is not None else output.float().mean()
                loss.backward()
                # Do not retain the graph between iterations.
                return loss.detach()
            finally:
                clear_gradients()

    return step


def module_wrapper(config, device, dtype):
    """Edit this function to construct your module and return a profiling step.

    The full config is available, including model/data and custom experiment keys.
    Allocate the model and inputs here so setup is excluded from measured time.
    Replace the module below with your own imported module. make_module_step
    handles train/eval mode, autograd, backward, and gradient cleanup. Pass a
    custom loss_fn to it if your module does not return a single floating tensor.
    """
    from implementation.module_dev import ModuleStack

    inputs = [torch.randn(*shape, device=device, dtype=dtype)
              for shape in config["experiment"]["input_shapes"]]
    if len(inputs) != 1:
        raise ValueError("The ModuleStack wrapper requires exactly one input shape")
    #module = MultiHeadLayerLL(4096, 32, device=device,dtype=dtype,use_rope=False)
    #module = PositionWiseFFLayer(4096, 12288, device=device, dtype=dtype)
    module = ModuleStack(4096, 4096, device=device, dtype=dtype)
    module.train(config["experiment"]["mode"] == "training")
    module.compile()
    return make_module_step(module, inputs, config["experiment"]["mode"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="YAML config path (default: %(default)s)")
    args = parser.parse_args()
    config = load_experiment_config(args.config_path)
    experiment = config["experiment"]
    device = cuda_device(experiment["device"])
    dtype = getattr(torch, experiment["dtype"])
    with torch.cuda.device(device):
        step = module_wrapper(config, device, dtype)
        result = profile_callable(step, device=device, warmup=experiment["warmup"],
                                  iterations=experiment["iterations"])
    result["mode"] = experiment["mode"]
    result["config_path"] = str(args.config_path.resolve())
    result["experiment"] = experiment
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()