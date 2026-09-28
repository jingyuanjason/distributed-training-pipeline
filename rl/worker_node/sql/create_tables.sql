CREATE TABLE prompts (
    id          bigint PRIMARY KEY,
    prompt_text text NOT NULL,
    priority    integer NOT NULL DEFAULT 0
);

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

    -- One rollout per prompt for this run and policy.
    UNIQUE (run_id, policy_version, prompt_id)
);