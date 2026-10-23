#!/usr/bin/env python3
"""
Evaluate a checkpoint on alpaca_eval / arena-hard / mt_bench with ArmoRM.

Steps:
  1. Load the model checkpoint and generate responses.
  2. Score each (prompt, response) with ArmoRM-Llama3-8B-v0.1.
  3. Save per-sample scores and a summary to the output directory.

MT-Bench follows the official protocol:
  - Turn 1: generate from [user1], score [user1, asst1]
  - Turn 2: generate from [user1, asst1, user2], score full 4-turn context
  - Final score per question = (turn1 + turn2) / 2

Usage:
  python3 eval/helpsteer2/eval_armo_reward.py \
      --model /path/to/checkpoint \
      --reward_model /path/to/ArmoRM-Llama3-8B-v0.1 \
      --datasets alpaca_eval arena_hard mt_bench \
      --output_dir /path/to/output \
      --batch_size 8
"""

import argparse
import json
import os
import statistics



# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get(
    "HELPSTEER_DATA_ROOT",
    os.path.join(REPO_ROOT, "data", "helpsteer2"),
)

ALPACA_EVAL_PATH = os.environ.get(
    "ALPACA_EVAL_PATH",
    os.path.join(DATA_ROOT, "alpaca_eval.json"),
)
ARENA_HARD_PATH = os.environ.get(
    "ARENA_HARD_PATH",
    os.path.join(DATA_ROOT, "arena_hard_v0.1.jsonl"),
)
MT_BENCH_PATH = os.environ.get(
    "MT_BENCH_PATH",
    os.path.join(
        DATA_ROOT,
        "mt_bench.parquet",
    ),
)


# ---------------------------------------------------------------------------
# Dataset loaders
#
# alpaca_eval / arena_hard:
#   returns List[{"id", "dataset", "messages": [user turns only]}]
#
# mt_bench:
#   returns List[{"id", "dataset", "user_turns": [str, str]}]
#   (special key; handled separately in generation)
# ---------------------------------------------------------------------------

def load_alpaca_eval():
    data = json.load(open(ALPACA_EVAL_PATH))
    return [
        {
            "id": f"alpaca_{i}",
            "dataset": "alpaca_eval",
            "messages": [{"role": "user", "content": item["instruction"]}],
        }
        for i, item in enumerate(data)
    ]


def load_arena_hard():
    samples = []
    with open(ARENA_HARD_PATH) as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            samples.append({"id": item["uid"], "dataset": "arena_hard", "messages": messages})
    return samples


def load_mt_bench():
    import pandas as pd
    df = pd.read_parquet(MT_BENCH_PATH)
    seen = set()
    samples = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid in seen:
            continue
        seen.add(qid)
        conv = row["conversation_a"]
        user_turns = [m["content"] for m in conv if m["role"] == "user"]
        samples.append({
            "id": f"mt_bench_{qid}",
            "dataset": "mt_bench",
            "user_turns": user_turns,  # [turn1_question, turn2_question]
        })
    return samples


DATASET_LOADERS = {
    "alpaca_eval": load_alpaca_eval,
    "arena_hard":  load_arena_hard,
    "mt_bench":    load_mt_bench,
}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _vllm_generate(llm, tokenizer, message_lists, max_new_tokens):
    """Generate responses for a list of message lists using vLLM."""
    if not message_lists:
        return []
    from vllm import SamplingParams
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in message_lists
    ]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def generate_responses(samples, model_path, max_new_tokens):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    print(f"[gen] Loading model with vLLM: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", 10240)),
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.9)),
        tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", 4)),
    )

    # Split by dataset type
    single_turn = [s for s in samples if s["dataset"] != "mt_bench"]
    mt_bench    = [s for s in samples if s["dataset"] == "mt_bench"]

    results = []

    # --- Single-turn datasets (alpaca_eval, arena_hard) ---
    print(f"[gen] Generating single-turn ({len(single_turn)} samples)...")
    responses = _vllm_generate(llm, tokenizer, [s["messages"] for s in single_turn], max_new_tokens)
    for s, resp in zip(single_turn, responses):
        results.append({**s, "response": resp})

    # --- MT-Bench: two-pass generation ---
    print(f"[gen] Generating mt_bench turn1 ({len(mt_bench)} samples)...")
    turn1_msgs = [[{"role": "user", "content": s["user_turns"][0]}] for s in mt_bench]
    asst1_list = _vllm_generate(llm, tokenizer, turn1_msgs, max_new_tokens)

    print(f"[gen] Generating mt_bench turn2 ({len(mt_bench)} samples)...")
    turn2_msgs = [
        [
            {"role": "user",      "content": s["user_turns"][0]},
            {"role": "assistant", "content": asst1},
            {"role": "user",      "content": s["user_turns"][1]},
        ]
        for s, asst1 in zip(mt_bench, asst1_list)
    ]
    asst2_list = _vllm_generate(llm, tokenizer, turn2_msgs, max_new_tokens)

    for s, asst1, asst2 in zip(mt_bench, asst1_list, asst2_list):
        results.append({
            "id": s["id"] + "_turn1",
            "dataset": "mt_bench",
            "mt_bench_qid": s["id"],
            "messages": [{"role": "user", "content": s["user_turns"][0]}],
            "response": asst1,
        })
        results.append({
            "id": s["id"] + "_turn2",
            "dataset": "mt_bench",
            "mt_bench_qid": s["id"],
            "messages": [
                {"role": "user",      "content": s["user_turns"][0]},
                {"role": "assistant", "content": asst1},
                {"role": "user",      "content": s["user_turns"][1]},
            ],
            "response": asst2,
        })

    del llm
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Scoring with ArmoRM
# ---------------------------------------------------------------------------

ARMO_SCORE_DIMS = (
    ("score_helpfulness", 0),
    ("score_correctness", 1),
    ("score_coherence", 2),
    ("score_complexity", 3),
    ("score_verbosity", 4),
)


def score_with_armo(samples, reward_model_path, batch_size, reward_dims=3):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"[score] Loading ArmoRM: {reward_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()

    scored = []
    for i in tqdm(range(0, len(samples), batch_size), desc="scoring"):
        batch = samples[i: i + batch_size]
        convs = [
            list(s["messages"]) + [{"role": "assistant", "content": s["response"]}]
            for s in batch
        ]
        texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in convs]
        tokenizer.truncation_side = "left"
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=8192,
        ).to(model.device)
        tokenizer.truncation_side = "right"

        with torch.inference_mode():
            rewards = model(**encoded).rewards.float()  # [B, 19]

        for j, s in enumerate(batch):
            item = {**s}
            for name, idx in ARMO_SCORE_DIMS[:reward_dims]:
                item[name] = rewards[j, idx].item()
            total3 = (rewards[j, 0] + rewards[j, 1] + rewards[j, 2]).item()
            if reward_dims == 5:
                item["score_total_3"] = total3
                item["score_total_5"] = sum(rewards[j, idx] for _, idx in ARMO_SCORE_DIMS).item()
            item["score_total"] = item["score_total_5"] if reward_dims == 5 else total3
            scored.append(item)

    del model
    torch.cuda.empty_cache()
    return scored


def compute_summary(scored, datasets, reward_dims=3):
    summary = {}
    score_keys = [name for name, _ in ARMO_SCORE_DIMS[:reward_dims]]
    if reward_dims == 5:
        score_keys += ["score_total_3", "score_total_5"]
    score_keys += ["score_total"]

    for ds in datasets:
        ds_scores = [s for s in scored if s["dataset"] == ds]
        if not ds_scores:
            continue

        summary[ds] = {}
        if ds == "mt_bench":
            from collections import defaultdict
            qid_dim = {dim: defaultdict(list) for dim in score_keys}
            for s in ds_scores:
                for dim in qid_dim:
                    qid_dim[dim][s["mt_bench_qid"]].append(s[dim])
            for dim in qid_dim:
                vals = [statistics.mean(v) for v in qid_dim[dim].values()]
                summary[ds][dim] = _stats(vals)
        else:
            for dim in score_keys:
                vals = [s[dim] for s in ds_scores]
                summary[ds][dim] = _stats(vals)

    return summary


def _stats(vals):
    return {
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_dims", type=int, choices=[3, 5], default=3)
    p.add_argument("--model", required=True)
    p.add_argument("--reward_model", required=True)
    p.add_argument("--datasets", nargs="+", default=["alpaca_eval", "arena_hard", "mt_bench"],
                   choices=list(DATASET_LOADERS.keys()))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--skip_generate", action="store_true",
                   help="Skip generation and reuse existing responses.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    responses_path = os.path.join(args.output_dir, "responses.jsonl")
    scores_path    = os.path.join(args.output_dir, "scores.jsonl")
    summary_path   = os.path.join(args.output_dir, "summary.json")

    # Generate
    if args.skip_generate:
        if not os.path.isfile(responses_path):
            raise FileNotFoundError(f"--skip_generate requires {responses_path}")
        print(f"[gen] Reusing {responses_path}")
        with open(responses_path) as f:
            all_responses = [json.loads(l) for l in f if l.strip()]
        all_responses = [r for r in all_responses if r['dataset'] in args.datasets]
    else:
        all_samples = []
        for ds in args.datasets:
            samples = DATASET_LOADERS[ds]()
            print(f"[data] {ds}: {len(samples)} questions")
            all_samples.extend(samples)
        all_responses = generate_responses(
            all_samples, args.model, args.max_new_tokens
        )
        with open(responses_path, "w") as f:
            for r in all_responses:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[gen] Saved {len(all_responses)} responses to {responses_path}")

    # Score
    scored = score_with_armo(all_responses, args.reward_model, args.batch_size, reward_dims=args.reward_dims)
    with open(scores_path, "w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[score] Saved {len(scored)} scores to {scores_path}")

    # Summary
    summary = compute_summary(scored, args.datasets, args.reward_dims)
    summary["score_dims"] = args.reward_dims
    summary["model"] = args.model
    summary["total_scored"] = len(scored)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
