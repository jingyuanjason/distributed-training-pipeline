"""Prompt loading for RL worker nodes."""

from pathlib import Path
from typing import Any, Self

import psycopg
import yaml
from psycopg.rows import dict_row

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_source.yaml"

_REQUIRED_DB_KEYS = ("host", "dbname", "user", "password")


class PromptLoader:
    def __init__(self):
        pass

    def samplePrompt(self) -> dict[str, Any] | None:
        pass


class DatabasePromptLoader(PromptLoader):
    """Loads prompts from a PostgreSQL database.

    Connection details and the prompt-fetching SQL are read from a YAML
    config file (see ``configs/data_source.example.yaml``). Each sampled
    prompt is returned as a dict keyed by the query's column names, e.g.::

        {"id": 1, "prompt_text": "...", "priority": 0}

    Returns ``None`` when the query yields no rows.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        super().__init__()
        self._config = self._load_config(config_path)
        self._conninfo = self._build_conninfo(self._config["database"])
        self._prompt_query = self._config["prompt_query"]
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

        if not config.get("prompt_query"):
            raise ValueError("Config must contain a non-empty 'prompt_query' SQL string.")
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

    def samplePrompt(self) -> dict[str, Any] | None:
        """Run the configured SQL and return the first row as a dict."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(self._prompt_query)
            row = cur.fetchone()
        return dict(row) if row is not None else None

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
