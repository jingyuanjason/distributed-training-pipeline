SELECT run_id, policy_version
FROM runs
WHERE status = 'active'
ORDER BY created_at DESC
LIMIT 1;
