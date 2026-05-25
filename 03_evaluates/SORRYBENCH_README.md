# SORRY-Bench sweep for the PsychoSafe paper

Runs four PsychoSafe-relevant Qwen3.5 checkpoints on SORRY-Bench (base + all
20 linguistic mutations × 3 system-prompt settings) and produces the table
for the paper.

## Models × system prompts

| Tag prefix      | Model path                          | Role                       |
|-----------------|-------------------------------------|----------------------------|
| `qwen35-a3b`    | `unsloth/Qwen3.5-35B-A3B`           | MoE base (35B / 3B active) |
| `qwen35-27b`    | `unsloth/Qwen3.5-27B`               | Dense base (27B)           |

Each tag is suffixed with `_default` (no sysprompt), `_v0` (`"You are a
helpful assistant"`), or `_v1` (the PsychoSafe system prompt at
`01_prompting/systemprompt-v1.txt`) — 12 model-prompt configs total.

Both psysafe SFT repos were uploaded without their `preprocessor_config.json`
(plus `processor_config.json`, `video_preprocessor_config.json`, and a few
other small JSONs that the Qwen3.5 multimodal arch needs to instantiate).
The `setup` stage copies those files from each SFT's parent base into the
local HF cache and clears the stale `.no_exist` markers before vLLM loads.

## Run on the 4× B200 node

```bash
export HF_TOKEN=hf_xxx       # needs access to sorry-bench/sorry-bench-202503
bash 03_evaluates/run_sorrybench.sh
```

That's it. The script is idempotent — re-running picks up where it left off.

The script runs five stages: `setup,gen,decode,judge,aggregate`. To rerun just
one (e.g. after fixing a bug in the aggregator):

```bash
STAGES=aggregate bash 03_evaluates/run_sorrybench.sh
```

Other knobs (env vars):

| Var               | Default                         | Notes                                  |
|-------------------|---------------------------------|----------------------------------------|
| `SORRY_DIR`       | `$HOME/sorry-bench`             | where sorry-bench is cloned/installed  |
| `PSYSAFE_DIR`     | this repo                       | auto-detected                          |
| `VLLM_VERSION`    | empty (latest)                  | pin a specific vLLM if needed          |
| `TENSOR_PARALLEL` | `4`                             | vLLM TP size — drop to 1 for testing   |
| `STAGES`          | `setup,gen,decode,judge,aggregate` | subset to run                       |

## Outputs

* Per-config generations, with `<think>…</think>` stripped:
  `${SORRY_DIR}/data/sorry_bench/model_answer{_<mut>}/<tag>.jsonl`
* Full generations including reasoning traces:
  `${SORRY_DIR}/data/sorry_bench/model_answer{_<mut>}/<tag>.with_thinking.jsonl`
* Per-mutation judge scores:
  `${SORRY_DIR}/data/sorry_bench/model_judgment{_<mut>}/ft-mistral-7b-instruct-v0.2.jsonl`
* Aggregated paper artifacts in `03_evaluates/results/sorrybench/`:
  * `sorrybench_overall.tex` — main paper table
  * `sorrybench_per_mutation.csv` — appendix detail
  * `sorrybench_per_category.csv` — per-category breakdown

## Smoke test (optional, ~2 min on 1 B200)

```bash
TENSOR_PARALLEL=1 STAGES=setup bash 03_evaluates/run_sorrybench.sh
cd "${SORRY_DIR:-$HOME/sorry-bench}"
source .venv/bin/activate
python gen_model_answer_vllm_psysafe.py \
    --model-path unsloth/Qwen3.5-35B-A3B \
    --tag-prefix smoke \
    --data-mutations question \
    --system-prompts default=none \
    --question-end 4 \
    --tensor-parallel-size 1 \
    --max-new-tokens 128
ls data/sorry_bench/model_answer/smoke_default*
```
