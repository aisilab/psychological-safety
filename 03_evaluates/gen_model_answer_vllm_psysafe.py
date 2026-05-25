"""Generate SORRY-Bench answers for the PsychoSafe paper.

Sibling of the upstream `gen_model_answer_vllm.py`. Differences:
  * Loads a model once and sweeps every (system_prompt x mutation) combo.
  * Uses the model's own HuggingFace chat template (Qwen3.5 chat template is
    not registered in FastChat).
  * Supports a `none` system prompt (no system message at all) plus arbitrary
    files (e.g. "You are a helpful assistant", PsychoSafe v1).
  * Strips `<think>…</think>` from the generated response before writing the
    judge-facing answer file; keeps the full output in `.with_thinking.jsonl`.
  * Forces the text-only causal-LM architecture via `hf_overrides` so vLLM
    skips the Qwen3.5 multimodal processor (and stops requiring the SFT repos
    to ship a `preprocessor_config.json`).

Output layout — matches what `gen_judgment_safety_vllm.py` expects:
    data/sorry_bench/model_answer{_<mut>}/<tag-prefix>_<sysprompt-key>.jsonl

Usage:
    python gen_model_answer_vllm_psysafe.py \\
        --model-path unsloth/Qwen3.5-35B-A3B \\
        --tag-prefix qwen35-a3b \\
        --data-mutations question,slang,role_play,translate-fr,... \\
        --system-prompts default=none,v0=prompts/systemprompt-v0.txt,v1=prompts/systemprompt-v1.txt

    # The system-prompts list is parsed as `key=path` or just `path`
    # (key inferred from the filename stem). The literal `none` means
    # "no system message at all".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import shortuuid
from transformers import AutoConfig
from vllm import LLM, SamplingParams

from common import load_questions  # upstream sorry-bench helper

THINK_END = "</think>"

# Preprocessor / tokenizer JSONs that the psysafe SFT repos were uploaded
# without. transformers' image-processor loader will refuse to load the model
# until these are present, even when vLLM is using a text-only causal-LM
# architecture via `hf_overrides`. We copy them from the parent base's HF
# snapshot before vLLM touches the model.
PATCH_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
)


def resolve_model_to_local_dir(model_path: str, parent_model_id: str | None) -> str:
    """Resolve an HF repo id to a local snapshot dir, patching missing
    preprocessor / processor JSONs from the parent base when needed.

    Returns the path to pass to vLLM. When `model_path` is already a local
    absolute directory, returns it unchanged. Passing a local directory to
    vLLM bypasses HF's cache-lookup machinery — which only recognises
    content-addressable blobs symlinked from `snapshots/<rev>/` and otherwise
    falls back to the hub (and creates `.no_exist` markers when files are
    missing there). Plain files we copy into `snapshots/<rev>/` are invisible
    to that machinery; they only become visible when transformers is reading
    a local directory directly.
    """
    if Path(model_path).is_absolute() and Path(model_path).exists():
        return model_path
    from huggingface_hub import snapshot_download

    print(f"[resolve] {model_path}: downloading snapshot (no-op if cached)")
    local_dir = snapshot_download(model_path)
    if parent_model_id:
        print(f"[resolve] {model_path}: copying preprocessor JSONs from {parent_model_id}")
        parent_dir = snapshot_download(
            parent_model_id, allow_patterns=["*.json", "*.txt"]
        )
        for fname in PATCH_FILES:
            src = os.path.join(parent_dir, fname)
            dst = os.path.join(local_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
                print(f"[resolve]   [copy] {fname}")
    print(f"[resolve] using local model dir: {local_dir}")
    return local_dir


def text_only_arch(model_path: str) -> tuple[str, bool]:
    """Return (text-only causal-LM class, is_moe) for a Qwen3.5 config.

    Qwen3.5 ships as a multimodal arch (`Qwen3_5{Moe,}ForConditionalGeneration`).
    Forcing vLLM to the matching text-only causal-LM class via `hf_overrides`
    avoids the vision-processor load path entirely.

    Note: must target the Qwen3_5* classes (not the older Qwen3*). transformers
    5.x represents Qwen3.5 with `Qwen3_5TextConfig` / `Qwen3_5MoeTextConfig`,
    which drop `max_window_layers` / `decoder_sparse_step`; vLLM's older Qwen3
    classes still assert on those attributes and crash on load.
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    is_moe = bool(getattr(text_cfg, "num_experts", None))
    return ("Qwen3_5MoeForCausalLM" if is_moe else "Qwen3_5ForCausalLM"), is_moe


def build_reasoning_splitter(parser_name: str | None, tokenizer):
    """Return a callable text -> (reasoning, answer).

    Prefer vLLM's registered parser (handles edge cases like Qwen3.5's
    `<tool_call>` implicit end-of-thinking); fall back to a manual `</think>`
    split if the parser isn't available.
    """
    if parser_name and parser_name.lower() != "none":
        try:
            from vllm.reasoning import ReasoningParserManager

            parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
            parser = parser_cls(tokenizer)
            print(f"Using vLLM reasoning parser: {parser_name}")

            def _split(text: str) -> tuple[str, str]:
                reasoning, content = parser.extract_reasoning(text, request=None)
                return (reasoning or "").strip(), (content or "").strip()

            return _split
        except Exception as exc:
            print(
                f"!! reasoning parser '{parser_name}' unavailable ({exc}); "
                f"falling back to manual </think> split"
            )

    def _manual(text: str) -> tuple[str, str]:
        if THINK_END not in text:
            return "", text.strip()
        reasoning, answer = text.split(THINK_END, 1)
        return reasoning.replace("<think>", "", 1).strip(), answer.strip()

    return _manual


def parse_sysprompt_arg(spec: str) -> tuple[str, str | None]:
    """Parse one `--system-prompts` entry. Returns (key, content-or-None)."""
    spec = spec.strip()
    if "=" in spec:
        key, path = spec.split("=", 1)
        key = key.strip()
    else:
        path = spec
        key = "default" if path == "none" else re.sub(
            r"^systemprompt[-_]?", "", Path(path).stem
        )
    if path == "none":
        return key, None
    return key, Path(path).read_text(encoding="utf-8").strip()


def build_messages(question: dict, system_prompt: str | None) -> list[dict]:
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question["turns"][0]})
    return msgs


def answer_paths(
    bench_name: str, mutation: str | None, model_id: str
) -> tuple[Path, Path]:
    base = "model_answer" if mutation in (None, "", "question") else f"model_answer_{mutation}"
    base_dir = Path(f"data/{bench_name}/{base}")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{model_id}.jsonl", base_dir / f"{model_id}.with_thinking.jsonl"


def already_done(answer_file: Path, question_ids: set[int]) -> set[int]:
    if not answer_file.exists():
        return set()
    done = set()
    with answer_file.open() as fin:
        for line in fin:
            try:
                done.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done & question_ids


def write_answer_rows(
    answer_file: Path,
    with_thinking_file: Path,
    questions: list[dict],
    outputs,
    model_id: str,
    split_fn,
) -> None:
    with answer_file.open("a") as out, with_thinking_file.open("a") as out_full:
        for q, vllm_output in zip(questions, outputs):
            full_turns, stripped_turns = [], []
            for choice in vllm_output.outputs:
                full = choice.text.strip()
                _, stripped = split_fn(full)
                full_turns.append(full)
                # If reasoning never closed (truncated mid-thought), keep the
                # full text so we never emit an empty answer to the judge.
                stripped_turns.append(stripped or full)
            tstamp = time.time()
            answer_id = shortuuid.uuid()
            base_row = {
                "question_id": q["question_id"],
                "answer_id": answer_id,
                "model_id": model_id,
                "tstamp": tstamp,
            }
            out.write(json.dumps(
                {**base_row, "choices": [{"index": 0, "turns": stripped_turns}]},
                ensure_ascii=False,
            ) + "\n")
            out_full.write(json.dumps(
                {**base_row, "choices": [{"index": 0, "turns": full_turns}]},
                ensure_ascii=False,
            ) + "\n")


def reorg_answer_file(answer_file: Path) -> None:
    """Sort by question_id, drop duplicates (keep last)."""
    if not answer_file.exists():
        return
    by_qid: dict[int, str] = {}
    with answer_file.open() as fin:
        for line in fin:
            try:
                qid = json.loads(line)["question_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            by_qid[qid] = line if line.endswith("\n") else line + "\n"
    with answer_file.open("w") as fout:
        for qid in sorted(by_qid):
            fout.write(by_qid[qid])


def run_sweep(args: argparse.Namespace) -> None:
    sysprompts = [parse_sysprompt_arg(s) for s in args.system_prompts.split(",")]
    mutations = [m.strip() for m in args.data_mutations.split(",") if m.strip()]
    print(f"System prompts: {[k for k, _ in sysprompts]}")
    print(f"Mutations:      {mutations}")

    args.model_path = resolve_model_to_local_dir(args.model_path, args.parent_model_id)

    text_arch, is_moe = text_only_arch(args.model_path)
    print(f"[hf-overrides] forcing architecture: {text_arch}  (MoE={is_moe})")

    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=is_moe,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        hf_overrides={"architectures": [text_arch]},
    )
    split_fn = build_reasoning_splitter(args.reasoning_parser, llm.get_tokenizer())

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
    )
    chat_template_kwargs = {"enable_thinking": args.enable_thinking}

    for mutation in mutations:
        if mutation in ("", "question"):
            question_file = f"data/{args.bench_name}/question.jsonl"
            mutation_key: str | None = None
        else:
            question_file = f"data/{args.bench_name}/question_{mutation}.jsonl"
            mutation_key = mutation
        if not Path(question_file).exists():
            print(f"[skip] {question_file} missing")
            continue

        all_questions = load_questions(
            question_file, args.question_begin, args.question_end
        )
        all_qids = {q["question_id"] for q in all_questions}

        for sp_key, sp_text in sysprompts:
            model_id = f"{args.tag_prefix}_{sp_key}"
            answer_file, with_thinking_file = answer_paths(
                args.bench_name, mutation_key, model_id
            )
            done = already_done(answer_file, all_qids)
            todo = [q for q in all_questions if q["question_id"] not in done]
            if not todo:
                print(
                    f"[done] {model_id} / mut={mutation or 'question'} "
                    f"({len(all_qids)} qids cached)"
                )
                continue
            print(
                f"[gen ] {model_id} / mut={mutation or 'question'} "
                f"/ {len(todo)} new (of {len(all_qids)}) prompts"
            )

            messages = [build_messages(q, sp_text) for q in todo]
            t0 = time.time()
            outputs = llm.chat(
                messages,
                sampling_params=sampling_params,
                chat_template_kwargs=chat_template_kwargs,
                use_tqdm=True,
            )
            dt = time.time() - t0
            print(
                f"       {len(outputs)} generations in {dt:.1f}s "
                f"({len(outputs) / max(dt, 1e-6):.2f}/s)"
            )

            write_answer_rows(
                answer_file, with_thinking_file, todo, outputs, model_id, split_fn,
            )
            reorg_answer_file(answer_file)
            reorg_answer_file(with_thinking_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HuggingFace repo id or local path.")
    parser.add_argument(
        "--parent-model-id",
        default=None,
        help="Optional parent base model id. If set, missing preprocessor / "
        "processor JSONs are copied from the parent's HF snapshot into the "
        "model's snapshot before vLLM loads it. Required for SFT repos "
        "uploaded without preprocessor_config.json.",
    )
    parser.add_argument(
        "--tag-prefix",
        required=True,
        help="Short id used to name answer files: <tag-prefix>_<sysprompt>.jsonl",
    )
    parser.add_argument("--bench-name", default="sorry_bench")
    parser.add_argument(
        "--data-mutations",
        required=True,
        help="Comma-separated mutation suffixes. Use 'question' for the base set.",
    )
    parser.add_argument(
        "--system-prompts",
        required=True,
        help="Comma-separated list of system-prompt specs. Each spec is either "
        "'key=path', a bare 'path' (key inferred from filename stem), or "
        "the literal 'none' (no system message).",
    )
    parser.add_argument("--question-begin", type=int, default=None, help="Debug slice start.")
    parser.add_argument("--question-end", type=int, default=None, help="Debug slice end.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--reasoning-parser",
        default="qwen3",
        help="vLLM reasoning parser name (default: qwen3). "
        "Use 'none' to disable and fall back to manual </think> split.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="SORRY-Bench default.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="4×B200 default.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        help="Headroom for v1 sysprompt (~7K toks) + long mutated questions "
        "+ long thinking + answer.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--enable-thinking", action="store_true", default=True,
        help="Qwen3 chat-template flag. Default on; we strip <think>…</think> "
        "from the judge-facing file post-hoc.",
    )
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
