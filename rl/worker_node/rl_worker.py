"""RL rollout worker.

Each worker node runs one of these loops against the vLLM server attached to
it (see rl/llm_host/start_vllm.py). The loop continuously:

    1) checks liveness of the vLLM server (GET /health),
    2) samples a group of prompts from the database (DatabasePromptLoader),
    3) generates a group of rollouts per prompt via vLLM /v1/completions,
    4) saves the generated rollouts into the database, tagged with the
       model version reported by the vLLM server.

vLLM connection info and sampling parameters come from
``configs/vllm_config.yaml``; database credentials and the prompt query come
from ``configs/data_source.yaml`` (see configs/data_source.example.yaml).
"""

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Self

import psycopg
import yaml

from rl.worker_node.prompt_loader import DatabasePromptLoader

logger = logging.getLogger(__name__)

DEFAULT_VLLM_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "vllm_config.yaml"
)

_INSERT_ROLLOUT_SQL = """
    INSERT INTO rollouts (prompt_id, rollout_text, model_version, worker_id)
    VALUES (%(prompt_id)s, %(rollout_text)s, %(model_version)s, %(worker_id)s);
"""


class VLLMClient:
    """Thin HTTP client for the vLLM OpenAI-compatible server."""

    def __init__(self, config: dict[str, Any]):
        vllm_cfg = config["vllm"]
        self.base_url = f"http://{vllm_cfg['host']}:{vllm_cfg['port']}"
        self.model_id = vllm_cfg["model_id"]
        self.health_path = vllm_cfg.get("health_path", "/health")
        self.request_timeout = vllm_cfg.get("request_timeout_seconds", 300)
        self.health_timeout = vllm_cfg.get("health_timeout_seconds", 5)

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> Any:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body) if body else None

    def is_alive(self) -> bool:
        """True if the vLLM server answers its health endpoint with 200."""
        try:
            req = urllib.request.Request(self.base_url + self.health_path, method="GET")
            with urllib.request.urlopen(req, timeout=self.health_timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def get_model_version(self) -> str:
        """Model version tag = the served model id reported by /v1/models."""
        try:
            info = self._request("/v1/models", None, self.health_timeout)
            for model in info.get("data", []):
                if model.get("id") == self.model_id:
                    return model["id"]
            if info.get("data"):
                return info["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, TypeError):
            logger.warning("Could not query /v1/models; falling back to config model_id")
        return self.model_id

    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling_params: dict[str, Any],
    ) -> list[list[str]]:
        """Generate ``n`` completions per prompt; returns one list per prompt."""
        payload = {
            "model": self.model_id,
            "prompt": prompts,
            "n": n,
            **sampling_params,
        }
        response = self._request("/v1/completions", payload, self.request_timeout)
        rollouts: list[list[str]] = [[] for _ in prompts]
        for choice in response.get("choices", []):
            rollouts[choice["index"]].append(choice.get("text", ""))
        return rollouts



class RolloutWorker:
    """Continuously samples prompts and writes rollouts to the database."""

    def __init__(
        self,
        vllm_config_path: str | Path = DEFAULT_VLLM_CONFIG_PATH,
        data_source_config_path: str | Path | None = None,
        worker_id: str | None = None,
        run_id: str | None = None,
    ):
        with open(vllm_config_path) as f:
            config = yaml.safe_load(f)
        worker_cfg = config.get("worker", {})

        self.vllm = VLLMClient(config)
        self.sampling_params = dict(config.get("sampling", {}))
        self.prompt_group_size = worker_cfg.get("prompt_group_size", 8)
        self.rollouts_per_prompt = worker_cfg.get("rollouts_per_prompt", 4)
        self.retry_interval = worker_cfg.get("retry_interval_seconds", 10)
        self._model_version_override = worker_cfg.get("model_version") or None

        loader_kwargs = (
            {"config_path": data_source_config_path} if data_source_config_path else {}
        )
        self.prompt_loader = DatabasePromptLoader(**loader_kwargs)

        # Reuse the loader's conninfo for writing rollouts.
        self._conninfo = self.prompt_loader._conninfo
        self._conn: psycopg.Connection | None = None

        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"

    # -- database writes -----------------------------------------------------

    def _get_connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._conninfo, autocommit=True)
        return self._conn

    def save_rollouts(
        self,
        prompt_id: int,
        rollout_texts: list[str],
        model_version: str,
    ) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.executemany(
                _INSERT_ROLLOUT_SQL,
                [
                    {
                        "prompt_id": prompt_id,
                        "rollout_text": text,
                        "model_version": model_version,
                        "worker_id": self.worker_id,
                    }
                    for text in rollout_texts
                ],
            )

    # -- main loop -----------------------------------------------------------

    def _sample_prompt_group(
        self, run_id: str, policy_version: str
    ) -> list[dict[str, Any]]:
        prompts = []
        for _ in range(self.prompt_group_size):
            prompt = self.prompt_loader.samplePrompt(
                run_id=run_id,
                policy_version=policy_version,
                worker_id=self.worker_id,
            )
            if prompt is not None:
                prompts.append(prompt)
        return prompts

    def run_once(self) -> bool:
        """One iteration of the rollout loop. Returns True if work was done."""
        if not self.vllm.is_alive():
            logger.warning("vLLM server at %s is not alive", self.vllm.base_url)
            return False

        model_version = self._model_version_override or self.vllm.get_model_version()
        run_id = self.prompt_loader.get_active_run(policy_version=model_version)

        prompts = self._sample_prompt_group(run_id, model_version)
        if not prompts:
            logger.info("No prompts available")
            return False

        prompt_texts = [p["prompt_text"] for p in prompts]
        rollouts = self.vllm.generate(
            prompt_texts, self.rollouts_per_prompt, self.sampling_params
        )
        for prompt, rollout_texts in zip(prompts, rollouts):
            self.save_rollouts(prompt["prompt_id"], rollout_texts, model_version)
            if not self.prompt_loader.complete_job(prompt["job_id"], self.worker_id):
                logger.warning(
                    "Job %s for prompt %s was no longer leased to this worker",
                    prompt["job_id"],
                    prompt["prompt_id"],
                )
        logger.info(
            "Saved %d rollouts for %d prompts (run_id=%s, model_version=%s)",
            sum(len(r) for r in rollouts),
            len(prompts),
            run_id,
            model_version,
        )
        return True

    def run(self) -> None:
        logger.info("Starting rollout worker %s", self.worker_id)
        while True:
            try:
                did_work = self.run_once()
            except Exception:
                logger.exception("Rollout loop iteration failed")
                did_work = False
            if not did_work:
                time.sleep(self.retry_interval)

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
        self.prompt_loader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL rollout worker node")
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Unique id for this worker (default: <hostname>-<pid>)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    with RolloutWorker(worker_id=args.worker_id) as worker:
        worker.run()

