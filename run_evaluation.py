#!/usr/bin/env python3
"""FineVAU evaluation: an LVLM describes FineW3 videos, a judge scores the descriptions, FV-Score summarises them.

    python run_evaluation.py generate --video-dir /path/to/videos
    python run_evaluation.py judge
    python run_evaluation.py score
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import scoring
from prompts import JUDGE_PROMPT, MODEL_PROMPT

DATASET = "joao-cardeira/FineW3"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_JUDGE = "google/gemma-4-31B-it"

log = logging.getLogger("finevau")
_ENGINES = []


def track_engine(llm):
    _ENGINES.append(llm)
    return llm


def shutdown_engines():
    while _ENGINES:
        llm = _ENGINES.pop()
        shutdown = getattr(getattr(getattr(llm, "llm_engine", None), "engine_core", None), "shutdown", None)
        if not callable(shutdown):
            log.warning("could not shut the vLLM engine down")
            continue
        try:
            shutdown(timeout=30)
            log.info("vLLM engine shut down")
        except Exception as e:
            log.warning("vLLM engine shutdown failed: %s", e)


def install_signal_handlers():
    def handler(signum, _frame):
        print(f"\nreceived {signal.Signals(signum).name}, shutting vLLM down", file=sys.stderr, flush=True)
        shutdown_engines()
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handler)


atexit.register(shutdown_engines)


def load_llm(model, args):
    from vllm import LLM

    return track_engine(LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    ))


def load_finew3(args):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="test")
    if "all" not in args.sources:
        ds = ds.filter(lambda s: s["source"] in args.sources)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    return ds


def ground_truth_json(sample):
    gt = {k: json.loads(sample[k]) if sample[k] else None for k in ("events", "entities", "scene")}
    return {
        "Location": gt["scene"],
        "Entity": gt["entities"],
        "Event": gt["events"].get("refined_events", []) if gt["events"] else [],
    }


def build_judge_prompt(sample, answer):
    return JUDGE_PROMPT.format(ground_truth=json.dumps(ground_truth_json(sample), indent=2), model_answer=answer)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    rows = []
    for n, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if n != len(lines) - 1:
                raise ValueError(f"{path}: line {n + 1} is not valid JSON")
            log.warning("%s: ignoring an incomplete last line", path)
    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def prepare_video_input(processor, path, args):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": path, "fps": args.fps, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
            {"type": "text", "text": MODEL_PROMPT},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    _, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    request = {"prompt": prompt, "multi_modal_data": {"video": videos}, "mm_processor_kwargs": video_kwargs}
    return request, prompt.rstrip().endswith("<think>")


def cmd_generate(args):
    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import SamplingParams

    have = {}
    for r in read_jsonl(args.output):
        have.setdefault(r["video_id"], set()).add(r["sample"])
    todo = [s for s in load_finew3(args) if not have.get(s["video_id"], set()) >= set(range(args.k))]
    log.info("%d videos to describe, %d answers each", len(todo), args.k)
    if not todo:
        return 0

    params = SamplingParams(
        n=args.k, seed=args.seed, max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, min_p=args.min_p, presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = load_llm(args.model, args)
    try:
        for batch in tqdm(list(chunks(todo, args.batch_size)), desc="generate"):
            requests, ready = [], []
            for s in batch:
                path = os.path.join(args.video_dir, f"{s['video_id']}.mp4")
                if not os.path.exists(path):
                    log.warning("missing video %s", path)
                    continue
                try:
                    request, thinking = prepare_video_input(processor, path, args)
                except Exception as e:
                    log.warning("could not read %s: %s", path, e)
                    continue
                requests.append(request)
                ready.append((s, thinking))
            if not requests:
                continue
            rows = []
            for (s, thinking), out in zip(ready, llm.generate(requests, params)):
                for c in sorted(out.outputs, key=lambda c: c.index):
                    rows.append({
                        "video_id": s["video_id"], "sample": c.index, "source": s["source"],
                        "answer": scoring.strip_reasoning(c.text, thinking), "output": c.text, "thinking": thinking,
                        "finish_reason": str(c.finish_reason), "output_tokens": len(c.token_ids),
                    })
            append_jsonl(args.output, rows)
    finally:
        shutdown_engines()
    return 0


def make_chat_fn(args):
    from vllm import SamplingParams

    llm = load_llm(args.judge_model, args)
    tokenizer = llm.get_tokenizer()

    def chat(prompts, temperature):
        results = [("", "prompt_too_long")] * len(prompts)
        send, params = [], []
        for i, p in enumerate(prompts):
            budget = min(args.max_tokens, args.max_model_len - len(tokenizer.encode(p)) - 64)
            if budget >= 1024:
                send.append(i)
                params.append(SamplingParams(temperature=temperature, max_tokens=budget))
        if send:
            messages = [[{"role": "user", "content": prompts[i]}] for i in send]
            outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
            for i, o in zip(send, outputs):
                results[i] = (o.outputs[0].text, str(o.outputs[0].finish_reason))
        return results

    return chat


def judge_all(prompts, chat_fn, retries):
    results, pending = {}, list(prompts)
    for attempt in range(1, retries + 2):
        if not pending:
            break
        still = []
        for key, (text, finish) in zip(pending, chat_fn([prompts[k] for k in pending], 0.0)):
            raw = scoring.parse_reply(text)
            if raw is None:
                still.append(key)
                results[key] = {"scores": None, "counts": None, "attempts": attempt,
                                "error": f"unusable judge reply (finish_reason={finish})"}
            else:
                results[key] = {"scores": raw, "counts": scoring.recall_counts(raw), "attempts": attempt, "error": None}
        pending = still
    return results


def judge_videos(keys, samples, predictions, chat_fn, retries):
    rows, prompts = {}, {}
    for key in keys:
        answer = predictions[key]["answer"]
        if answer is None:
            rows[key] = {"status": "no_answer", "scores": None, "attempts": 0,
                         "counts": scoring.zero_counts(ground_truth_json(samples[key[0]])),
                         "error": "the reasoning did not finish"}
        else:
            prompts[key] = build_judge_prompt(samples[key[0]], answer)
    if prompts:
        for key, res in judge_all(prompts, chat_fn, retries).items():
            rows[key] = {"status": "judged" if res["scores"] is not None else "judge_failed", **res}
    return [{"video_id": k[0], "sample": k[1], "source": samples[k[0]]["source"], **rows[k]} for k in keys]


def cmd_judge(args):
    predictions = {(r["video_id"], r["sample"]): r for r in read_jsonl(args.predictions)}
    if not predictions:
        log.error("no predictions in %s, run generate first", args.predictions)
        return 1
    samples = {s["video_id"]: s for s in load_finew3(args)}
    keys = sorted(k for k in predictions if k[0] in samples)
    done = {(r["video_id"], r["sample"]) for r in read_jsonl(args.output) if r["status"] in ("judged", "no_answer")}
    todo = [k for k in keys if k not in done]
    to_judge = sum(1 for k in todo if predictions[k]["answer"] is not None)
    log.info("%d answers to score, %d of them need the judge", len(todo), to_judge)

    if todo:
        chat_fn = make_chat_fn(args) if to_judge else None
        try:
            for chunk in chunks(todo, args.chunk_size):
                append_jsonl(args.output, judge_videos(chunk, samples, predictions, chat_fn, args.retries))
        finally:
            shutdown_engines()
    return report(args, {k[0] for k in keys})


def report(args, video_ids=None):
    latest = {}
    for r in read_jsonl(args.output):
        if video_ids is None or r["video_id"] in video_ids:
            latest[(r["video_id"], r["sample"])] = r
    if not latest:
        log.error("no judgments in %s", args.output)
        return 1
    weights = {"what": args.lambda_what, "who": args.lambda_who, "where": args.lambda_where}
    agg = scoring.aggregate(latest.values(), weights)
    print(scoring.format_report(agg))
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return 0


def cmd_score(args):
    if "all" in args.sources:
        return report(args)
    return report(args, {r["video_id"] for r in read_jsonl(args.output) if r["source"] in args.sources})


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp, output):
        sp.add_argument("--dataset", default=DATASET)
        sp.add_argument("--sources", nargs="+", default=["UCA"], help="UCA, ECVA or all")
        sp.add_argument("--limit", type=int, help="only the first N videos")
        sp.add_argument("--output", default=output)

    def vllm_args(sp, max_model_len):
        sp.add_argument("--tensor-parallel-size", type=int, default=1)
        sp.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        sp.add_argument("--max-model-len", type=int, default=max_model_len)

    def score_args(sp):
        sp.add_argument("--lambda-what", type=float, default=scoring.PAPER_WEIGHTS["what"])
        sp.add_argument("--lambda-who", type=float, default=scoring.PAPER_WEIGHTS["who"])
        sp.add_argument("--lambda-where", type=float, default=scoring.PAPER_WEIGHTS["where"])
        sp.add_argument("--summary", default="fv_score.json")

    g = sub.add_parser("generate", help="the LVLM describes each video")
    data_args(g, "predictions.jsonl")
    vllm_args(g, 65536)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--video-dir", required=True, help="directory with {video_id}.mp4 files")
    g.add_argument("--fps", type=float, default=2.0)
    g.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    g.add_argument("--max-pixels", type=int, default=1638400)
    g.add_argument("--max-new-tokens", type=int, default=16384, help="reasoning and answer together")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--k", type=int, default=4, help="answers per video (mean@k)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--min-p", type=float, default=0.0)
    g.add_argument("--presence-penalty", type=float, default=1.5)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.set_defaults(func=cmd_generate)

    j = sub.add_parser("judge", help="the judge scores each answer and FV-Score is printed")
    data_args(j, "judgments.jsonl")
    vllm_args(j, 65536)
    score_args(j)
    j.add_argument("--predictions", default="predictions.jsonl")
    j.add_argument("--judge-model", default=DEFAULT_JUDGE)
    j.add_argument("--max-tokens", type=int, default=20480)
    j.add_argument("--retries", type=int, default=2)
    j.add_argument("--chunk-size", type=int, default=256)
    j.set_defaults(func=cmd_judge)

    s = sub.add_parser("score", help="FV-Score from a judgments file")
    s.add_argument("--output", default="judgments.jsonl")
    s.add_argument("--sources", nargs="+", default=["all"])
    score_args(s)
    s.set_defaults(func=cmd_score)
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    install_signal_handlers()
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
