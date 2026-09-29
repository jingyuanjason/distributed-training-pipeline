-- Rotates the active run: serializes against concurrent workers, marks the
-- old run exhausted, and inserts a fresh run. Must be executed inside a
-- transaction (the advisory lock is transaction-scoped). Readers acquire
-- the same lock before reading, so they block while a rotation is running.

SELECT pg_advisory_xact_lock(hashtext('rl_run_rotation'));

UPDATE runs
SET status = 'exhausted',
    completed_at = now()
WHERE run_id = $1;

INSERT INTO runs (run_id, policy_version)
VALUES ($2, $3);
