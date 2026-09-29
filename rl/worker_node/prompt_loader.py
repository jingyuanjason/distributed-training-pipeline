"""Prompt loading for RL worker nodes."""

import re
import uuid
from pathlib import Path
from typing import Any, Self

import psycopg
import yaml
from psycopg.rows import dict_row

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_source.yaml"

SQL_DIR = Path(__file__).resolve().parent / "sql"
BATCH_SAMPLE_SQL_PATH = SQL_DIR / "batch_sample.sql"
COMPLETE_JOB_SQL_PATH = SQL_DIR / "complete_job.sql"
ROTATE_RUN_SQL_PATH = SQL_DIR / "rotate_run.sql"
GET_ACTIVE_RUN_SQL_PATH = SQL_DIR / "get_active_run.sql"
COUNT_REMAINING_PROMPTS_SQL_PATH = SQL_DIR / "count_remaining_prompts.sql"
GET_PROMPT_SQL_PATH = SQL_DIR / "get_prompt.sql"

_REQUIRED_DB_KEYS = ("host", "dbname", "user", "password")

_PLACEHOLDER_RE = re.compile(r"\$(\d+)")


def _load_sql_statements(path: Path) -> list[str]:
    """Read a .sql file, rewrite ``$N`` placeholders as ``%(pN)s``, and split
    it into individual statements.

    Transaction-control statements (BEGIN/COMMIT) are stripped; callers wrap
    multi-statement scripts in ``Connection.transaction()`` instead.
    """
    sql = path.read_text()
    sql = _PLACEHOLDER_RE.sub(r"%(p\1)s", sql)
    return [
        stmt
        for stmt in (s.strip() for s in sql.split(";"))
        if stmt and stmt.upper() not in ("BEGIN", "COMMIT")
    ]


def _load_sql(path: Path) -> str:
    """Load a .sql file that must contain exactly one executable statement."""
    statements = _load_sql_statements(path)
    if len(statements) != 1:
        raise ValueError(f"Expected a single statement in {path}, got {len(statements)}")
    return statements[0]


_BATCH_SAMPLE_SQL = _load_sql(BATCH_SAMPLE_SQL_PATH)
_COMPLETE_JOB_SQL = _load_sql(COMPLETE_JOB_SQL_PATH)
_LOCK_RUN_ROTATION_SQL, _EXHAUST_RUN_SQL, _CREATE_RUN_SQL = _load_sql_statements(
    ROTATE_RUN_SQL_PATH
)
_GET_ACTIVE_RUN_SQL = _load_sql(GET_ACTIVE_RUN_SQL_PATH)
_COUNT_REMAINING_PROMPTS_SQL = _load_sql(COUNT_REMAINING_PROMPTS_SQL_PATH)
_GET_PROMPT_SQL = _load_sql(GET_PROMPT_SQL_PATH)


class PromptLoader:
    def __init__(self):
        pass

    def samplePrompt(
        self, run_id: str, policy_version: str, worker_id: str
    ) -> dict[str, Any] | None:
        pass

    def complete_job(self, job_id: int, worker_id: str) -> bool:
        pass

    def get_active_run(self, policy_version: str) -> str:
        pass


class DatabasePromptLoader(PromptLoader):
    """Loads prompts from a PostgreSQL database using the job-queue SQL.

    Connection details are read from a YAML config file (see
    ``configs/data_source.example.yaml``). Sampling leases a job via
    ``sql/batch_sample.sql`` (SKIP LOCKED, so concurrent workers never pick
    the same prompt); jobs are finished via ``sql/complete_job.sql``.

    Each sampled prompt is returned as a dict containing the leased
    ``rollout_jobs`` row plus the prompt fields, e.g.::

        {"job_id": 12, "run_id": "...", "prompt_id": 1, "policy_version": "v1",
         "status": "leased", "worker_id": "host-1", ..., "prompt_text": "..."}

    Returns ``None`` when no prompt is available.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        super().__init__()
        self._config = self._load_config(config_path)
        self._conninfo = self._build_conninfo(self._config["database"])
        self._conn: psycopg.Connection | None = None

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Data source config not found: {config_path}. "
                "Copy configs/data_source.example.yaml to configs/data_source.yaml "
                "and fill in your database credentials."
            )
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError(f"Invalid data source config: {config_path}")

        db = config.get("database")
        if not isinstance(db, dict):
            raise TypeError("Config must contain a 'database' mapping.")
        missing = [k for k in _REQUIRED_DB_KEYS if k not in db]
        if missing:
            raise ValueError(f"Config 'database' is missing required keys: {missing}")
        return config

    @staticmethod
    def _build_conninfo(db_config: dict[str, Any]) -> str:
        return " ".join(f"{key}={value}" for key, value in db_config.items())

    def _get_connection(self) -> psycopg.Connection:
        """Return a live connection, (re)connecting lazily as needed."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                self._conninfo,
                autocommit=True,
                row_factory=dict_row,
            )
        return self._conn

    def samplePrompt(
        self, run_id: str, policy_version: str, worker_id: str
    ) -> dict[str, Any] | None:
        """Lease one prompt via ``batch_sample.sql`` and return it.

        ``run_id`` identifies the training-run iteration, ``policy_version``
        the model/policy version being sampled for, and ``worker_id`` the
        worker taking the lease.
        """
        conn = self._get_connection()
        params = {"p1": run_id, "p2": policy_version, "p3": worker_id}
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(_BATCH_SAMPLE_SQL, params)
            job = cur.fetchone()
        if job is None:
            return None

        job = dict(job)
        with conn.cursor() as cur:
            cur.execute(_GET_PROMPT_SQL, {"p1": job["prompt_id"]})
            prompt = cur.fetchone()
        if prompt is None:
            return None
        result = {**job, **dict(prompt)}
        result["job_id"] = job["id"]
        return result

    def complete_job(self, job_id: int, worker_id: str) -> bool:
        """Mark a leased job as completed via ``complete_job.sql``.

        Returns True if the job was still leased to this worker and got
        marked completed.
        """
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(_COMPLETE_JOB_SQL, {"p1": job_id, "p2": worker_id})
            return cur.rowcount == 1

    def get_active_run(self, policy_version: str) -> str:
        """Return the run_id of the current active run, creating one if needed.

        If no active run exists, or every prompt already has a rollout job
        for the latest active run (regardless of the policy_version those
        jobs were leased under), the old run (if any) is marked exhausted
        and a fresh run is created with a new run_id.

        The transaction-scoped advisory lock from ``rotate_run.sql`` is
        acquired before reading, so a worker's read blocks while another
        worker is rotating the run, and rotations are serialized.
        """
        conn = self._get_connection()
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(_LOCK_RUN_ROTATION_SQL)
            cur.execute(_GET_ACTIVE_RUN_SQL)
            run = cur.fetchone()

            if run is not None:
                cur.execute(_COUNT_REMAINING_PROMPTS_SQL, {"p1": run["run_id"]})
                remaining = cur.fetchone()["remaining"]
                if remaining > 0:
                    return str(run["run_id"])
                cur.execute(_EXHAUST_RUN_SQL, {"p1": run["run_id"]})

            run_id = str(uuid.uuid4())
            cur.execute(_CREATE_RUN_SQL, {"p2": run_id, "p3": policy_version})
            return run_id

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
