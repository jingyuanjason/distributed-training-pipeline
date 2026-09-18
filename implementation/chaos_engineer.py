import time
import random
import logging
import os
import signal
import multiprocessing

import torch

logger = logging.getLogger(__name__)

# Holds references to leaked GPU tensors so they are never freed,
# mimicking a memory leak.
_leaked_tensors = []

def random_delay(delay_time: int = 5, chance: float = 0.01):
    if random.random() < chance:
        logger.info(f"[chaos] random_delay triggered: sleeping {delay_time}s")
        time.sleep(delay_time)

def random_kill(chance: float = 0.001):
    if random.random() < chance:
        logger.warning(f"[chaos] random_kill triggered: killing process {os.getpid()} with SIGKILL")
        os.kill(os.getpid(), signal.SIGKILL)

def random_gpu_memory_leak(size_mb: int = 512, chance: float = 0.01):
    if random.random() < chance:
        num_elements = size_mb * 1024 * 1024 // 4  # float32
        tensor = torch.empty(num_elements, dtype=torch.float32, device="cuda")
        _leaked_tensors.append(tensor)  # keep a reference so it is never freed
        logger.warning(
            f"[chaos] random_gpu_memory_leak triggered: leaked {size_mb}MB GPU tensor "
            f"({len(_leaked_tensors)} leaks, ~{len(_leaked_tensors) * size_mb}MB total)"
        )

def _cpu_burn(duration: float):
    """Busy-loop burning CPU for `duration` seconds. Runs in a worker process."""
    deadline = time.time() + duration
    x = 0
    while time.time() < deadline:
        x += 1

def random_cpu_hog(num_workers: int = 4, duration: float = 10, chance: float = 0.01):
    if random.random() < chance:
        workers = []
        for _ in range(num_workers):
            worker = multiprocessing.Process(target=_cpu_burn, args=(duration,), daemon=True)
            worker.start()
            workers.append(worker.pid)
        logger.warning(
            f"[chaos] random_cpu_hog triggered: spawned {num_workers} CPU-burning "
            f"processes (pids={workers}) for {duration}s"
        )

def function_register(chaos_function, parameters):
    def trigger():
        chaos_function(**parameters)
    return trigger

CHAOS_FUNCTIONS = {
    "random_delay": random_delay,
    "random_kill": random_kill,
    "random_gpu_memory_leak": random_gpu_memory_leak,
    "random_cpu_hog": random_cpu_hog,
}

class ChaosEnginnerTrigger():
    def __init__(self, chaos_config: dict):
        self._triggers = []
        for name, parameters in (chaos_config or {}).items():
            if not parameters.get("enabled", False):
                continue
            chaos_function = CHAOS_FUNCTIONS.get(name)
            if chaos_function is None:
                raise ValueError(f"Unknown chaos function: {name}")
            parameters = {k: v for k, v in parameters.items() if k != "enabled"}
            self._triggers.append(function_register(chaos_function, parameters))

    def trigger(self):
        for trigger in self._triggers:
            trigger()
