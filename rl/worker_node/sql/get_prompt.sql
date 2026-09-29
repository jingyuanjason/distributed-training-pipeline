SELECT id, prompt_text, priority
FROM prompts
WHERE id = $1;
