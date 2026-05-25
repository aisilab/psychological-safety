# LLM Judge

`llm_judge.py` uses an external judge model to score assistant responses against the rubric in `CRITERIA_llm.md`.

## Files

- `llm_judge.py`: runs the judging pipeline
- `CRITERIA_llm.md`: judging instructions given to the model
- `envs.env`: stores `CHAT_AI_TOKEN` and retry/timeout settings

## Run

From the repository root:

```bash
python 03_evaluates/llm_judge.py
```

## Expected Data

The default `main()` path reads paired evaluation files from `01_prompting/results/`:

- `val_v0_sft_splitted_with_category.json`
- `val_v1_sft_splitted_with_category.json`

Each file is expected to contain a `results` list with paired items that include at least:

- `prompt`
- `answer`
- `risk cluster`
- `risk category`

This can be generated using `01_prompting/run_benchmark_vllm.py` with the appropriate settings.

## Notes

- The current entry point is hardcoded in `main()`.
- The default flow writes JSON and Markdown outputs into `03_evaluates/output/`.
