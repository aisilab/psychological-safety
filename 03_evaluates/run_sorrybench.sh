#!/usr/bin/env bash
# run_sorrybench.sh — One-shot SORRY-Bench sweep for the PsychoSafe paper.
#
# Runs on a 4×B200 node inside the `pao` conda env. Idempotent: safe to re-run.
# Set HF_TOKEN in your env first (needs access to sorry-bench/sorry-bench-202503).
#
# Tunables via env vars:
#   SORRY_DIR        — sorry-bench checkout dir (default: /work/psychosafety/sorry-bench)
#   PSYSAFE_DIR      — this repo (auto-detected from script location)
#   CONDA_ENV        — conda env to install into (default: pao)
#   CONDA_BASE       — conda install root (default: $HOME/miniconda3)
#   VLLM_SPEC        — vLLM constraint (default: 'vllm>=0.21'). Uses uv's
#                      --torch-backend=auto so the right cu13x wheel is chosen.
#   MAX_MODEL_LEN    — vLLM max_model_len (default: 65536). v1 sysprompt is
#                      ~7K tokens; long persuasion/morse mutations push past 8K.
#   MAX_NEW_TOKENS   — generation cap (default: 8192). Qwen3.5 emits its
#                      reasoning inline (not in <think> tags) and 1024 was
#                      truncating before the actual answer appeared.
#   TENSOR_PARALLEL  — vLLM TP size (default: 4)
#   HF_HOME          — HF cache (default: /work/psychosafety/.hf_cache)
#   STAGES           — comma-separated: setup,gen,decode,judge,aggregate
#                      (default: all)
#
# Usage:
#   HF_TOKEN=hf_xxx bash 03_evaluates/run_sorrybench.sh
#   STAGES=judge,aggregate bash 03_evaluates/run_sorrybench.sh
set -euo pipefail

# ---------- locate repos ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PSYSAFE_DIR="${PSYSAFE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SORRY_DIR="${SORRY_DIR:-${PSYSAFE_DIR}/sorry-bench}"
CONDA_ENV="${CONDA_ENV:-pao}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.21}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
HF_HOME="${HF_HOME:-${PSYSAFE_DIR}/.hf_cache}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
STAGES="${STAGES:-setup,gen,decode,judge,aggregate}"

# ---------- conda env activation ----------
# Activate `pao` (CUDA 13.2 runtime + python 3.13). We install with `uv pip`
# into this env — never `conda install` (per the node owner's instruction).
activate_pao() {
  # conda's activate scripts touch unset vars (e.g. NVCC_PREPEND_FLAGS);
  # disable nounset around the activation, restore after.
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
uv_pip() { uv pip install --python "${CONDA_PREFIX}/bin/python" "$@"; }

# Models and prompt settings. Tag = short id used in answer/judge filenames.
# Each SFT entry also names its parent base — we copy the parent's preprocessor
# JSONs into the SFT cache before vLLM loads it (psysafe SFTs were uploaded
# without preprocessor_config.json / processor_config.json / video_preprocessor_config.json
# and Qwen3.5's multimodal arch refuses to instantiate without them).
MODELS=(
  "unsloth/Qwen3.5-35B-A3B|qwen35-a3b|"
  "unsloth/Qwen3.5-27B|qwen35-27b|"
)

# 20 mutations + base. `question` is the base 440-prompt set.
MUTATIONS=(
  question
  slang uncommon_dialects technical_terms role_play misspellings
  logical_appeal authority_endorsement misrepresentation
  evidence-based_persuasion expert_endorsement
  ascii caesar morse atbash
  translate-ml translate-ta translate-mr translate-zh-cn translate-fr
)
CIPHER_MUTS=(ascii caesar morse atbash)
TRANSLATE_MUTS=(translate-ml translate-ta translate-mr translate-zh-cn translate-fr)

run_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ---------- 1. setup ----------
if run_stage setup; then
  echo "==> [setup] sorry-bench checkout at ${SORRY_DIR}"
  if [[ ! -d "${SORRY_DIR}" ]]; then
    git clone https://github.com/SORRY-Bench/sorry-bench.git "${SORRY_DIR}"
  fi

  # Drop the patched generator and judge into the sorry-bench tree
  # (overwrites on re-run). The judge patch truncates degenerate-loop
  # responses so the 32K Mistral context isn't blown.
  cp "${SCRIPT_DIR}/gen_model_answer_vllm_psysafe.py" \
     "${SORRY_DIR}/gen_model_answer_vllm_psysafe.py"
  cp "${SCRIPT_DIR}/gen_judgment_safety_vllm.py" \
     "${SORRY_DIR}/gen_judgment_safety_vllm.py"

  # Copy the PsychoSafe v0/v1 prompts so the generator can read them locally.
  mkdir -p "${SORRY_DIR}/prompts"
  cp "${PSYSAFE_DIR}/01_prompting/systemprompt-v0.txt" "${SORRY_DIR}/prompts/"
  cp "${PSYSAFE_DIR}/01_prompting/systemprompt-v1.txt" "${SORRY_DIR}/prompts/"

  activate_pao
  echo "==> [setup] installing into conda env: ${CONDA_PREFIX}"

  # vLLM bundles its own torch wheel matched to the detected CUDA (cu130 here).
  # --torch-backend=auto tells uv to honor that and avoid pulling a conflicting
  # torch from PyPI.
  uv_pip --torch-backend=auto "${VLLM_SPEC}"
  uv_pip "fschat[model_worker,llm_judge]" shortuuid tqdm \
      "huggingface_hub[cli]" hf-transfer

  # Download SORRY-Bench question set (gated) and judge weights.
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "!! HF_TOKEN unset. Set it (must have access to sorry-bench/sorry-bench-202503) and re-run." >&2
    exit 1
  fi
  echo "==> [setup] HF cache: ${HF_HOME}"
  mkdir -p "${HF_HOME}"

  echo "==> [setup] downloading SORRY-Bench question set"
  mkdir -p "${SORRY_DIR}/data/sorry_bench"
  hf download sorry-bench/sorry-bench-202503 \
      --repo-type dataset \
      --local-dir "${SORRY_DIR}/data/sorry_bench" \
      --token "${HF_TOKEN}"

  echo "==> [setup] downloading safety judge weights"
  mkdir -p "${SORRY_DIR}/ckpts/finetuned_models"
  hf download sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406 \
      --local-dir "${SORRY_DIR}/ckpts/finetuned_models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406" \
      --token "${HF_TOKEN}"

  # Side uv venv for googletrans (incompatible with openai, hence isolated).
  # Install step runs every time so the venv tracks the pinned version even
  # after an upgrade (e.g. 3.1.0a0 → 4.0.2).
  if [[ ! -d "${SORRY_DIR}/.venv-googletrans" ]]; then
    echo "==> [setup] creating googletrans venv for translate decoding"
    (cd "${SORRY_DIR}" && uv venv --python 3.12 .venv-googletrans)
  fi
  uv pip install --python "${SORRY_DIR}/.venv-googletrans/bin/python" \
      "googletrans==4.0.2"

  # Patch SFT model caches with their parent's preprocessor files. The Qwen3.5
  # arch is multimodal (`Qwen3_5{Moe,}ForConditionalGeneration`), so vLLM tries
  # to build a vision processor on load; the psysafe SFT repos were uploaded
  # without those JSONs. This step is idempotent — re-running it is a no-op
  # once the files are in place.
  echo "==> [setup] patching psysafe SFT caches with parent preprocessor files"
  PATCH_PAIRS_PY=""
  for entry in "${MODELS[@]}"; do
    PARENT="$(awk -F'|' '{print $3}' <<<"$entry")"
    SFT="$(awk -F'|' '{print $1}' <<<"$entry")"
    [[ -n "$PARENT" ]] && PATCH_PAIRS_PY+="    (\"${SFT}\", \"${PARENT}\"),"$'\n'
  done
  export PATCH_PAIRS_PY HF_TOKEN
  python - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
PATCH_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
]
pairs_src = os.environ.get("PATCH_PAIRS_PY", "").strip()
PAIRS = []
exec("PAIRS = [\n" + pairs_src + "\n]")
token = os.environ.get("HF_TOKEN") or None
for sft, parent in PAIRS:
    print(f"[patch] {sft}  <-  {parent}")
    # Just the small text/json files of the parent — no weights.
    parent_dir = snapshot_download(
        parent,
        allow_patterns=["*.json", "*.txt"],
        token=token,
    )
    # And the small files of the SFT (so its snapshot dir exists).
    sft_dir = snapshot_download(
        sft,
        allow_patterns=["*.json", "*.jinja"],
        token=token,
    )
    for fname in PATCH_FILES:
        src = os.path.join(parent_dir, fname)
        dst = os.path.join(sft_dir, fname)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst):
            print(f"        [skip ] {fname}")
        else:
            shutil.copy(src, dst)
            print(f"        [copy ] {fname}")
    # Clear stale .no_exist markers that hf_hub may have written when a
    # previous load probed for these files and missed.
    no_exist_root = os.path.join(
        os.path.dirname(os.path.dirname(sft_dir)), ".no_exist"
    )
    if os.path.isdir(no_exist_root):
        for revision in os.listdir(no_exist_root):
            for fname in PATCH_FILES:
                marker = os.path.join(no_exist_root, revision, fname)
                if os.path.exists(marker):
                    os.remove(marker)
                    print(f"        [unmark] {fname}")
PY
fi

# ---------- 2. generate answers ----------
if run_stage gen; then
  activate_pao
  cd "${SORRY_DIR}"
  MUT_LIST="$(IFS=, ; echo "${MUTATIONS[*]}")"
  SP_LIST="default=none,v0=prompts/systemprompt-v0.txt,v1=prompts/systemprompt-v1.txt"

  for entry in "${MODELS[@]}"; do
    MODEL_PATH="$(awk -F'|' '{print $1}' <<<"$entry")"
    TAG_PREFIX="$(awk -F'|' '{print $2}' <<<"$entry")"
    PARENT_ID="$(awk -F'|' '{print $3}' <<<"$entry")"
    PARENT_FLAG=()
    [[ -n "${PARENT_ID}" ]] && PARENT_FLAG=(--parent-model-id "${PARENT_ID}")
    echo "==> [gen ] ${MODEL_PATH} → tag=${TAG_PREFIX} (parent=${PARENT_ID:-none})"
    python gen_model_answer_vllm_psysafe.py \
        --model-path "${MODEL_PATH}" \
        --tag-prefix "${TAG_PREFIX}" \
        "${PARENT_FLAG[@]}" \
        --bench-name sorry_bench \
        --data-mutations "${MUT_LIST}" \
        --system-prompts "${SP_LIST}" \
        --tensor-parallel-size "${TENSOR_PARALLEL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature 0.7
  done
fi

# ---------- 3. decode cipher + translate mutations ----------
if run_stage decode; then
  cd "${SORRY_DIR}/data/sorry_bench/mutate"

  activate_pao
  echo "==> [dec ] decoding cipher mutations (ascii/caesar/morse/atbash)"
  python - <<'PY'
import copy, json, os
from encode_experts import encode_expert_dict
for method in ["ascii", "caesar", "morse", "atbash"]:
    expert = encode_expert_dict[method]
    model_answer_dir = f"../model_answer_{method}"
    if not os.path.isdir(model_answer_dir):
        continue
    for f in os.listdir(model_answer_dir):
        if not f.endswith(".jsonl") or f.endswith("_decoded.jsonl"):
            continue
        if f.endswith(".with_thinking.jsonl"):
            continue
        out = f"{model_answer_dir}/{f.split('.jsonl')[0]}_decoded.jsonl"
        if os.path.exists(out):
            print(f"[skip] {out}")
            continue
        print(f"[decode] {model_answer_dir}/{f}")
        with open(out, "w") as outfile, open(f"{model_answer_dir}/{f}") as infile:
            for line in infile:
                row = json.loads(line)
                d = copy.deepcopy(row)
                d["choices"][0]["turns"][0] = expert.decode(row["choices"][0]["turns"][0])
                outfile.write(json.dumps(d) + "\n")
PY

  echo "==> [dec ] translating non-English mutations (5 langs)"
  # googletrans 4.x is async and intermittently hits transient TLS/HTTP-2
  # write errors against translate.googleapis.com from cloud IPs. We retry
  # each row with exponential backoff, stream output to a .partial sidecar,
  # and only rename to the final name once the file is fully translated —
  # so a mid-file crash is fully resumable on rerun.
  "${SORRY_DIR}/.venv-googletrans/bin/python" - <<'PY'
import asyncio, copy, json, os, re
from googletrans import Translator

MAX_RETRIES = 6
METHODS = ["translate-ml", "translate-ta", "translate-mr", "translate-zh-cn", "translate-fr"]
# Source-script Unicode blocks for the repair pass. googletrans 3.1.0a0 used
# to silently return the *input* text on transport failure, leaving rows in
# the source script even though they were written to disk as if translated.
# We detect that by counting source-script chars in the translated text and
# retranslating any row above BAD_THRESHOLD. French (Latin script) has no
# reliable single-character heuristic, so it's omitted — the new 4.0.2 path
# raises on failure rather than silently returning input, so fr never enters
# this state in fresh runs anyway.
LANG_SCRIPTS = {
    "ml":    re.compile(r"[ഀ-ൿ]"),  # Malayalam
    "ta":    re.compile(r"[஀-௿]"),  # Tamil
    "mr":    re.compile(r"[ऀ-ॿ]"),  # Devanagari
    "zh-cn": re.compile(r"[一-鿿]"),  # CJK Unified Ideographs
}
BAD_THRESHOLD = 10


# googletrans 4.x sends the source text inside the GET query string. httpx
# refuses URLs whose query component exceeds 65535 bytes — and once
# URL-encoded, a Malayalam char (3 UTF-8 bytes) is ~9 chars, so anything past
# ~7K Malayalam chars trips InvalidURL. SORRY-Bench rows go up to 23K chars,
# so we pre-chunk along newline boundaries before translating.
CHUNK_CHARS = 2000


def chunk_text(text, max_chars=CHUNK_CHARS):
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), max_chars):
                chunks.append(line[i:i + max_chars])
            continue
        candidate = line if not current else current + "\n" + line
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


async def translate_atomic(translator, text, src):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            res = await translator.translate(text, dest="en", src=src)
            return res.text
        except Exception as e:
            last_exc = e
            wait = min(60, 2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {type(e).__name__}: {e!s:.140s} — sleeping {wait}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"giving up after {MAX_RETRIES} retries; last error: {last_exc!r}")


async def translate_row(translator, text, src):
    chunks = chunk_text(text)
    if len(chunks) == 1:
        return await translate_atomic(translator, chunks[0], src)
    parts = []
    for chunk in chunks:
        parts.append(await translate_atomic(translator, chunk, src))
    return "\n".join(parts)


async def translate_file(translator, in_path, out_path, src):
    tmp = out_path + ".partial"
    done = 0
    if os.path.exists(tmp):
        with open(tmp) as g:
            done = sum(1 for _ in g)
        if done:
            print(f"  [resume] {os.path.basename(in_path)}: {done} rows already translated")
    with open(in_path) as infile:
        rows = [json.loads(line) for line in infile]
    if done >= len(rows):
        os.rename(tmp, out_path)
        return
    with open(tmp, "a") as outfile:
        for i in range(done, len(rows)):
            row = rows[i]
            text = row["choices"][0]["turns"][0]
            translated = await translate_row(translator, text, src)
            d = copy.deepcopy(row)
            d["choices"][0]["turns"][0] = translated
            outfile.write(json.dumps(d) + "\n")
            outfile.flush()
    os.rename(tmp, out_path)


async def repair_file(translator, in_path, out_path, src):
    """Patch rows where the old googletrans silently returned the input.
    Detected by source-script char count > BAD_THRESHOLD in the translated
    text. Writes a .repair sidecar then renames atomically. No-op for langs
    without a script-based heuristic (e.g. fr) or files with no bad rows."""
    script_re = LANG_SCRIPTS.get(src)
    if script_re is None:
        return
    with open(in_path) as g:
        in_rows = [json.loads(line) for line in g]
    with open(out_path) as g:
        out_rows = [json.loads(line) for line in g]
    if len(in_rows) != len(out_rows):
        # Shouldn't happen — translate_file() only renames .partial to out
        # when fully done. Bail loudly rather than corrupt.
        raise RuntimeError(
            f"repair: row-count mismatch in={len(in_rows)} out={len(out_rows)} for {out_path}"
        )
    bad_idx = [
        i for i, r in enumerate(out_rows)
        if len(script_re.findall(r["choices"][0]["turns"][0])) > BAD_THRESHOLD
    ]
    if not bad_idx:
        return
    print(f"  [repair {src}] {os.path.basename(out_path)}: {len(bad_idx)} silent-fail rows — patching")
    for j, i in enumerate(bad_idx):
        text = in_rows[i]["choices"][0]["turns"][0]
        translated = await translate_row(translator, text, src)
        out_rows[i] = copy.deepcopy(out_rows[i])
        out_rows[i]["choices"][0]["turns"][0] = translated
        if (j + 1) % 20 == 0:
            print(f"    [repair {src}] {os.path.basename(out_path)}: {j + 1}/{len(bad_idx)}")
    repair_tmp = out_path + ".repair"
    with open(repair_tmp, "w") as g:
        for r in out_rows:
            g.write(json.dumps(r) + "\n")
    os.replace(repair_tmp, out_path)


async def main():
    translator = Translator()
    for method in METHODS:
        model_answer_dir = f"../model_answer_{method}"
        if not os.path.isdir(model_answer_dir):
            continue
        src = method.split("translate-")[1]
        for f in sorted(os.listdir(model_answer_dir)):
            if not f.endswith(".jsonl"):
                continue
            if f.endswith("_translated_to_en.jsonl") or f.endswith(".with_thinking.jsonl"):
                continue
            in_path = f"{model_answer_dir}/{f}"
            out = f"{model_answer_dir}/{f.split('.jsonl')[0]}_translated_to_en.jsonl"
            need_translate = True
            if os.path.exists(out):
                with open(in_path) as g:
                    n_in = sum(1 for _ in g)
                with open(out) as g:
                    n_out = sum(1 for _ in g)
                if n_out == n_in:
                    print(f"[have] {out} (full {n_in} rows)")
                    need_translate = False
                else:
                    # Salvage existing rows via the .partial sidecar so
                    # translate_file() resumes from row n_out instead of redoing.
                    tmp = out + ".partial"
                    if os.path.exists(tmp):
                        with open(tmp) as g:
                            n_tmp = sum(1 for _ in g)
                        if n_tmp >= n_out:
                            print(f"[resume] {out} ({n_out}/{n_in}) — .partial already at {n_tmp}, dropping stale {os.path.basename(out)}")
                            os.remove(out)
                        else:
                            print(f"[resume] {out} ({n_out}/{n_in}) — promoting over shorter .partial ({n_tmp})")
                            os.remove(tmp)
                            os.rename(out, tmp)
                    else:
                        print(f"[resume] {out} ({n_out}/{n_in} rows) — renaming to .partial to continue")
                        os.rename(out, tmp)
            if need_translate:
                print(f"[translate {src}] {in_path}")
                await translate_file(translator, in_path, out, src)
            await repair_file(translator, in_path, out, src)


asyncio.run(main())
PY
fi

# ---------- 4. safety judge ----------
if run_stage judge; then
  activate_pao
  cd "${SORRY_DIR}"

  # Model IDs that the judge should evaluate (suffixes match generator output).
  MODEL_IDS=()
  for entry in "${MODELS[@]}"; do
    TAG_PREFIX="$(awk -F'|' '{print $2}' <<<"$entry")"
    for sp in default v0 v1; do
      MODEL_IDS+=("${TAG_PREFIX}_${sp}")
    done
  done
  echo "==> [judge] model ids: ${MODEL_IDS[*]}"

  for mut in "${MUTATIONS[@]}"; do
    if [[ "$mut" == "question" ]]; then
      judge_file="data/sorry_bench/model_judgment/ft-mistral-7b-instruct-v0.2.jsonl"
      mut_flag=()
    else
      judge_file="data/sorry_bench/model_judgment_${mut}/ft-mistral-7b-instruct-v0.2.jsonl"
      mut_flag=(--data-mutation "$mut")
    fi
    if [[ -f "$judge_file" ]]; then
      echo "[skip] judge already done for mut=${mut} (delete $judge_file to redo)"
      continue
    fi
    echo "==> [judge] mut=${mut}"
    python gen_judgment_safety_vllm.py "${mut_flag[@]}" \
        --judge-model ft-mistral-7b-instruct-v0.2 \
        --model-list "${MODEL_IDS[@]}"
  done
fi

# ---------- 5. aggregate for the paper ----------
if run_stage aggregate; then
  activate_pao
  cd "${PSYSAFE_DIR}"
  python 03_evaluates/sorrybench_aggregate.py \
      --sorry-bench-dir "${SORRY_DIR}" \
      --out-dir "${PSYSAFE_DIR}/03_evaluates/results/sorrybench"
  echo "==> results in ${PSYSAFE_DIR}/03_evaluates/results/sorrybench"
fi

echo "==> done."
