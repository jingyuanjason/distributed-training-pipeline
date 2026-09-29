BEGIN;

WITH candidate AS (
    SELECT p.id
    FROM prompts p
    WHERE NOT EXISTS (
        SELECT 1
        FROM rollout_jobs j
        WHERE j.prompt_id = p.id
          AND j.run_id = $1
    )
    ORDER BY p.priority DESC, p.id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
INSERT INTO rollout_jobs (
    run_id,
    prompt_id,
    policy_version,
    status,
    worker_id,
    lease_expires_at
)
SELECT
    $1,
    id,
    $2,
    'leased',
    $3,
    now() + interval '10 minutes'
FROM candidate
RETURNING *;

COMMIT;