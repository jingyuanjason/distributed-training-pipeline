UPDATE rollout_jobs
SET status = 'completed',
    completed_at = now()
WHERE id = $1
  AND worker_id = $2
  AND status = 'leased';