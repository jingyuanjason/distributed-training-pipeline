CREATE TABLE prompts (
    id          bigint PRIMARY KEY,
    prompt_text text NOT NULL,
    priority    integer NOT NULL DEFAULT 0
);

CREATE TABLE runs (
    run_id          uuid PRIMARY KEY,
    policy_version  text NOT NULL,
    status          text NOT NULL DEFAULT 'active', -- active, exhausted
    created_at      timestamptz NOT NULL DEFAULT now(),
    completed_at    timestamptz
);

CREATE INDEX runs_status_idx ON runs (status, created_at DESC);

CREATE TABLE rollout_jobs (
    id              bigserial PRIMARY KEY,
    run_id          uuid NOT NULL,
    prompt_id       bigint NOT NULL REFERENCES prompts(id),
    policy_version  text NOT NULL,

    status          text NOT NULL, -- leased, completed, failed
    worker_id       text NOT NULL,
    sampled_at      timestamptz NOT NULL DEFAULT now(),
    lease_expires_at timestamptz NOT NULL,
    completed_at    timestamptz,

    -- One rollout job per prompt within a run, regardless of policy version.
    UNIQUE (run_id, prompt_id)
);

CREATE TABLE rollouts (
    id              bigserial PRIMARY KEY,
    prompt_id       bigint NOT NULL REFERENCES prompts(id),
    rollout_text    text NOT NULL,
    model_version   text NOT NULL,
    worker_id       text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX rollouts_model_version_idx ON rollouts (model_version);
CREATE INDEX rollouts_prompt_id_idx ON rollouts (prompt_id);