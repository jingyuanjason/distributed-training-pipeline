SELECT count(*) AS remaining
FROM prompts p
WHERE NOT EXISTS (
    SELECT 1
    FROM rollout_jobs j
    WHERE j.prompt_id = p.id
      AND j.run_id = $1
);
