#!/usr/bin/env bash
# run_xstest.sh — Over-refusal companion to run_sorrybench.sh.
#
# Evaluates the same 4 checkpoints under the same 3 system prompts on
# XSTest~\citep{rottger2024xstest}: 250 safe prompts that look unsafe and
# should be answered, plus 200 contrast prompts that are unsafe and should be
# refused. The judge is a small open-weight LLM (default
# `Qwen/Qwen2.5-7B-Instruct`) with the standard XSTest 3-class prompt; lower
# safe-set refusal = less over-refusal, higher unsafe-set refusal = safer.
#
# Designed to run on a 1×B200 node inside the `pao` conda env. Idempotent.
# Share SORRY_DIR with run_sorrybench.sh so the SFT/parent model weights, the
# prompts/ directory and the HF cache are all reused.
#
# Tunables (env vars):
#   SORRY_DIR        — sorry-bench checkout (default: $PSYSAFE_DIR/sorry-bench)
#   PSYSAFE_DIR      — this repo (auto-detected)
#   CONDA_ENV        — conda env (default: pao)
#   CONDA_BASE       — conda root  (default: $HOME/miniconda3)
#   HF_TOKEN         — required if pulling gated models
#   TENSOR_PARALLEL  — vLLM TP size (default: 1 — half/single B200)
#   JUDGE_MODEL      — HF id of the XSTest judge (default Qwen2.5-7B-Instruct)
#   MAX_NEW_TOKENS   — generation cap (default 4096)
#   ALLOW_TAGS       — comma-separated subset of model tags to actually run
#                      (default: all four). Set e.g. to
#                      "qwen35-27b,psysafe-27b" to skip the MoE variants.
#   STAGES           — comma-separated: setup,gen,judge,aggregate (default all)
#
# Usage:
#   HF_TOKEN=hf_xxx bash 03_evaluates/run_xstest.sh
#   STAGES=judge,aggregate bash 03_evaluates/run_xstest.sh
set -euo pipefail

# ---------- locate repos ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PSYSAFE_DIR="${PSYSAFE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SORRY_DIR="${SORRY_DIR:-${PSYSAFE_DIR}/sorry-bench}"
CONDA_ENV="${CONDA_ENV:-pao}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ALLOW_TAGS="${ALLOW_TAGS:-qwen35-a3b,qwen35-27b,psysafe-a3b,psysafe-27b}"
HF_HOME="${HF_HOME:-${PSYSAFE_DIR}/.hf_cache}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
STAGES="${STAGES:-setup,gen,judge,aggregate}"

XSTEST_DIR="${SORRY_DIR}/data/xstest"
RESULTS_DIR="${SCRIPT_DIR}/results/xstest"

# ---------- conda env activation ----------
activate_pao() {
  set +u
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "!! conda activate ${CONDA_ENV} failed (CONDA_PREFIX empty)" >&2
    exit 1
  fi
}

# Same model list as run_sorrybench.sh. Entry: "<hf-id>|<tag>|<parent-id>".
MODELS=(
  "unsloth/Qwen3.5-35B-A3B|qwen35-a3b|"
  "unsloth/Qwen3.5-27B|qwen35-27b|"
  "lgalke/Qwen3.5-35B-A3B-psysafe|psysafe-a3b|unsloth/Qwen3.5-35B-A3B"
  "giannor/Qwen3.5-27B-psysafe|psysafe-27b|unsloth/Qwen3.5-27B"
)

# System prompts — keys must match run_sorrybench.sh so model_answer file
# names and downstream aggregations stay in sync.
SYSPROMPT_ARG="default=none,v0=${SORRY_DIR}/prompts/systemprompt-v0.txt,v1=${SORRY_DIR}/prompts/systemprompt-v1.txt"
SP_KEYS=(default v0 v1)

run_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
allow_tag() { [[ ",${ALLOW_TAGS}," == *",$1,"* ]]; }

# ---------- 1. setup ----------
if run_stage setup; then
  activate_pao
  echo "==> [setup] downloading XSTest CSV into ${XSTEST_DIR}/question.jsonl"
  python "${SCRIPT_DIR}/gen_xstest_questions.py" \
    --out "${XSTEST_DIR}/question.jsonl"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "==> [setup] prefetching judge weights ${JUDGE_MODEL}"
    hf download "${JUDGE_MODEL}" --token "${HF_TOKEN}" >/dev/null
  else
    echo "==> [setup] HF_TOKEN unset — judge prefetch skipped"
  fi

  # Sanity-check the v0/v1 prompts exist (they're written by run_sorrybench.sh
  # setup). If they don't, copy them straight from the PsychoSafe repo.
  mkdir -p "${SORRY_DIR}/prompts"
  for src in systemprompt-v0.txt systemprompt-v1.txt; do
    if [[ ! -f "${SORRY_DIR}/prompts/${src}" ]]; then
      cp "${PSYSAFE_DIR}/01_prompting/${src}" "${SORRY_DIR}/prompts/${src}"
    fi
  done
fi

# ---------- 2. gen ----------
if run_stage gen; then
  activate_pao
  cd "${SORRY_DIR}"

  for entry in "${MODELS[@]}"; do
    IFS="|" read -r model_path tag parent_id <<< "$entry"
    if ! allow_tag "$tag"; then
      echo "==> [skip] ${tag} (not in ALLOW_TAGS=${ALLOW_TAGS})"
      continue
    fi
    parent_flag=()
    [[ -n "$parent_id" ]] && parent_flag=(--parent-model-id "$parent_id")
    echo "==> [gen] ${tag}  (TP=${TENSOR_PARALLEL})"
    python gen_model_answer_vllm_psysafe.py \
        --model-path "${model_path}" \
        "${parent_flag[@]}" \
        --tag-prefix "${tag}" \
        --bench-name xstest \
        --data-mutations question \
        --system-prompts "${SYSPROMPT_ARG}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"
  done
fi

# ---------- 3. judge ----------
if run_stage judge; then
  activate_pao
  cd "${SORRY_DIR}"

  IDS=()
  for entry in "${MODELS[@]}"; do
    IFS="|" read -r _model_path tag _parent_id <<< "$entry"
    allow_tag "$tag" || continue
    for sp in "${SP_KEYS[@]}"; do
      IDS+=("${tag}_${sp}")
    done
  done
  IDS_CSV=$(IFS=,; echo "${IDS[*]}")
  echo "==> [judge] ${JUDGE_MODEL} on ${#IDS[@]} (model,sysprompt) pairs"

  python "${SCRIPT_DIR}/judge_xstest_vllm.py" \
      --bench-dir "${XSTEST_DIR}" \
      --model-ids "${IDS_CSV}" \
      --judge-model "${JUDGE_MODEL}" \
      --tensor-parallel-size "${TENSOR_PARALLEL}"
fi

# ---------- 4. aggregate ----------
if run_stage aggregate; then
  activate_pao
  cd "${PSYSAFE_DIR}"
  mkdir -p "${RESULTS_DIR}"
  python "${SCRIPT_DIR}/xstest_aggregate.py" \
      --bench-dir "${XSTEST_DIR}" \
      --out-dir "${RESULTS_DIR}"
  echo "==> [aggregate] wrote results to ${RESULTS_DIR}"
fi
