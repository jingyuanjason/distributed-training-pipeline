import logging
import os
import subprocess
import json
import time
logger = logging.getLogger(__name__)

def kill_existing_vllm_server(port: int) -> None:
    pattern = f"vllm serve .* --port {port}"
    try:
        result = subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
        if result.returncode == 0:
            time.sleep(2)
            subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    except FileNotFoundError:
        pass

def start_server(
    model_id: str,
    host: str,
    port: int,
    gpu: int,
    seed: int,
    load_format: str,
    logging_level: str,
    gpu_memory_utilization: float = 0.9,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_LOGGING_LEVEL"] = logging_level
    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        "1",
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
        "--load-format",
        load_format,
    ]
    logger.info("Starting vLLM server: %s", " ".join(command))
    return subprocess.Popen(command, env=env, start_new_session=True)

if __name__ == "__main__":
    kill_existing_vllm_server(8088)
    process = start_server("allenai/OLMo-2-0425-1B", "localhost", 8088, 0, seed=0, load_format="auto", logging_level="INFO")
    stdout, stderr = process.communicate()
    print("Return code:", process.returncode)
    print("stdout:", stdout)
    print("stderr:", stderr)