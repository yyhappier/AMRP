#!/usr/bin/env python3
"""Unified math evaluation: training prompt, vLLM generation, grading and summaries.

Normal mode runs the complete pipeline. --eval-only summarizes existing results.
The answer-equivalence implementation remains in oat_math_grader.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

# Generation dependencies are imported only when needed.
TRAIN_SYSTEM = (
    "You are an expert math solver. You should first think about the reasoning process "
    "in the mind and then provide the user with the answer. The reasoning process and "
    "answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, "
    "respectively, i.e., <reasoning> reasoning process here </reasoning>"
    "<answer> answer here </answer>"
)

def training_messages(prompt):
    """Replace/prepend the system prompt without changing dataset records."""
    messages = prompt.tolist() if hasattr(prompt, 'tolist') else list(prompt)
    system = {'role': 'system', 'content': TRAIN_SYSTEM}
    if messages and messages[0].get('role') == 'system':
        return [system] + messages[1:]
    return [system] + messages


def oat_evaluate(model_response, gt_answer, fast=False):
    from oat_math_grader import boxed_reward_fn, answer_tag_reward_fn_for_orz
    # 优先用 answer_tag_reward_fn_for_orz（支持 <answer> 标签）
    # 若无 <answer> 标签则回退到 boxed_reward_fn（支持 \boxed{}）
    if '<answer>' in model_response and '</answer>' in model_response:
        return answer_tag_reward_fn_for_orz(model_response, gt_answer, fast=fast)
    return boxed_reward_fn(model_response, gt_answer, fast=fast)

def grading_text(prompt, text):
    """Preserve the legacy handling of a tokenizer-provided think prefix."""
    if prompt.endswith('<think>\n'):
        text = '<think>\n' + text
        if '</think>' in text:
            text = text.split('</think>')[1]
    return text


def generate_and_grade(input_file, output_file, model_path,
                       temperature=0.0, top_p=1.0, max_tokens=3000, seed=42):
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_parquet(input_file)
    if df.empty:
        raise ValueError('Evaluation dataset is empty')
    messages = [training_messages(prompt) for prompt in df['prompt']]
    answers = [item['ground_truth'] for item in df['reward_model']]
    outputs = generate_vllm(messages, model_path, temperature, top_p, max_tokens, seed)
    if len(outputs) != len(answers) or any(len(out.outputs) != 1 for out in outputs):
        raise ValueError('Expected exactly one response per dataset row')

    decoded_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'
    with open(decoded_path, 'w') as stream:
        for out, answer in zip(outputs, answers):
            stream.write(json.dumps({'prompt': out.prompt,
                                     'generated_text': out.outputs[0].text,
                                     'answer': answer}) + '\n')

    records = []
    for out, answer in tqdm(zip(outputs, answers), total=len(answers), desc='Scoring'):
        text = grading_text(out.prompt, out.outputs[0].text)
        correct = oat_evaluate(text, answer, fast=False)[1] == 1.0
        records.append({'prompt': out.prompt, 'generated_text': text,
                        'answer': answer, 'correctness': correct})
    # Do not swallow output errors and accidentally summarize a stale result file.
    with open(output_file, 'w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')


def generate_vllm(messages, model_path, temperature=0.0, top_p=1.0, max_tokens=3000, seed=42):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    #vllm模型加载
    # 修复 tokenizer_config.json 中 extra_special_tokens 为 list 的兼容性问题：
    # 新版 transformers 要求该字段为 dict，旧版 checkpoint 保存为 list 时会报
    # AttributeError: 'list' object has no attribute 'keys'
    # 从 list 改为 {tok: tok} 的 dict 语义完全等价，不影响推理结果
    import json as _json, os as _os
    _tcfg_path = _os.path.join(model_path, 'tokenizer_config.json')
    if _os.path.exists(_tcfg_path):
        with open(_tcfg_path, 'r') as _f:
            _tcfg = _json.load(_f)
        if isinstance(_tcfg.get('extra_special_tokens'), list):
            _tcfg['extra_special_tokens'] = {t: t for t in _tcfg['extra_special_tokens']}
            with open(_tcfg_path, 'w') as _f:
                _json.dump(_tcfg, _f, ensure_ascii=False, indent=2)
            print(f"[FIX] extra_special_tokens: list -> dict in {_tcfg_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # max_tokens is for the maximum length for generation.
    print("tokens: ", max_tokens)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=1)
    print(torch.cuda.device_count())
    # 允许通过环境变量 VLLM_MAX_MODEL_LEN 限制上下文长度，避免 Qwen3 默认 262144 触发 KV cache OOM
    _vllm_kwargs = dict(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.85, seed=seed, enforce_eager=True)
    _vmml = os.environ.get('VLLM_MAX_MODEL_LEN')
    if _vmml:
        _vllm_kwargs['max_model_len'] = int(_vmml)
        print(f"[vllm] max_model_len override -> {_vmml}")
    llm = LLM(**_vllm_kwargs)  # enforce_eager=True 禁用 torch.compile，避免跨机器缓存 FileNotFoundError

    prompts = [tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) for message in messages]
    print('Example input:', prompts[0])
    return llm.generate(prompts, sampling_params)


# Metrics and report output

# Aligned with swift/plugin/orm.py::PureReasoningFormat
_FORMAT_RE = re.compile(
    r'^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$',
    re.DOTALL,
)


def _format_ok(text: str) -> bool:
    return bool(_FORMAT_RE.match(text or ''))


def _soft_overlong_reward(length: int, soft_max_length: int, soft_cache_length: int) -> float:
    expected = soft_max_length - soft_cache_length
    exceed = length - expected
    penalty = min(-exceed / soft_cache_length, 0)
    return max(1.0 + penalty, 0.0)


def summarize(
    jsonl_path: str,
    parquet_path: str,
    decoded_path: str | None = None,
    tokenizer_path: str | None = None,
    soft_max_length: int = 3000,
    soft_cache_length: int = 200,
) -> dict:
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=['data_source'])
    sources = df['data_source'].tolist()

    with open(jsonl_path) as f:
        results = [json.loads(line) for line in f]

    if len(results) != len(sources):
        raise ValueError(f"row count mismatch: jsonl={len(results)} parquet={len(sources)}")

    # Optional per-sample generated_text (for format + length).
    decoded_texts: list[str] | None = None
    if decoded_path:
        with open(decoded_path) as f:
            decoded_texts = [json.loads(line)['generated_text'] for line in f]
        if len(decoded_texts) != len(sources):
            raise ValueError(
                f"decoded row count mismatch: decoded={len(decoded_texts)} parquet={len(sources)}"
            )

    elif results and all("generated_text" in rec for rec in results):
        decoded_texts = [rec["generated_text"] for rec in results]

    # Optional tokenizer for token-level length.
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Bucket per data_source.
    buckets: dict[str, dict] = {}
    for i, (src, rec) in enumerate(zip(sources, results)):
        b = buckets.setdefault(src, {
            'correct': [], 'format': [], 'lengths': [], 'soft_overlong': [],
        })
        b['correct'].append(bool(rec['correctness']))
        if decoded_texts is not None:
            text = decoded_texts[i]
            b['format'].append(_format_ok(text))
            if tokenizer is not None:
                # add_special_tokens=False to count only the generated content
                n = len(tokenizer.encode(text, add_special_tokens=False))
                b['lengths'].append(n)
                b['soft_overlong'].append(
                    _soft_overlong_reward(n, soft_max_length, soft_cache_length)
                )

    def _summ_lengths(ls: list[int]) -> dict:
        if not ls:
            return {}
        ls_sorted = sorted(ls)
        return {
            'mean': sum(ls) / len(ls),
            'p50': ls_sorted[len(ls) // 2],
            'p95': ls_sorted[min(len(ls) - 1, int(0.95 * len(ls)))],
            'max': ls_sorted[-1],
            'trunc_rate': sum(1 for x in ls if x >= soft_max_length) / len(ls),
        }

    per_dataset = {}
    for src, b in buckets.items():
        total = len(b['correct'])
        correct = sum(b['correct'])
        entry = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total else 0.0,
        }
        if b['format']:
            entry['format_pass'] = sum(b['format'])
            entry['format_rate'] = entry['format_pass'] / total
        if b['lengths']:
            entry['length'] = _summ_lengths(b['lengths'])
            entry['soft_overlong_mean'] = sum(b['soft_overlong']) / total
        per_dataset[src] = entry

    all_correct = [c for b in buckets.values() for c in b['correct']]
    overall_total = len(all_correct)
    overall_correct = sum(all_correct)
    overall_micro = overall_correct / overall_total if overall_total else 0.0
    overall_macro = (
        sum(v['accuracy'] for v in per_dataset.values()) / len(per_dataset)
        if per_dataset else 0.0
    )
    overall = {
        'total': overall_total,
        'correct': overall_correct,
        'accuracy_micro': overall_micro,
        'accuracy_macro': overall_macro,
    }

    all_format = [f for b in buckets.values() for f in b['format']]
    if all_format:
        overall['format_pass'] = sum(all_format)
        overall['format_rate_micro'] = sum(all_format) / len(all_format)
        overall['format_rate_macro'] = (
            sum(v['format_rate'] for v in per_dataset.values()) / len(per_dataset)
        )

    all_lengths = [l for b in buckets.values() for l in b['lengths']]
    all_sol = [r for b in buckets.values() for r in b['soft_overlong']]
    if all_lengths:
        overall['length'] = _summ_lengths(all_lengths)
        overall['soft_overlong_mean_micro'] = sum(all_sol) / len(all_sol)
        overall['soft_overlong_mean_macro'] = (
            sum(v['soft_overlong_mean'] for v in per_dataset.values()) / len(per_dataset)
        )
        overall['soft_overlong_config'] = {
            'soft_max_length': soft_max_length,
            'soft_cache_length': soft_cache_length,
        }

    return {'per_dataset': per_dataset, 'overall': overall}


def write_outputs(summary: dict, out_dir: str, run_name: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'summary.json')
    csv_path = os.path.join(out_dir, 'summary.csv')

    payload = {'run': run_name, **summary}
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    order = ['math', 'amc', 'aime']
    known = [d for d in order if d in summary['per_dataset']]
    extras = [d for d in summary['per_dataset'] if d not in order]
    datasets = known + extras

    has_format = any('format_rate' in summary['per_dataset'][d] for d in datasets)
    has_length = any('length' in summary['per_dataset'][d] for d in datasets)

    header = ['run', 'dataset', 'correct', 'total', 'accuracy']
    if has_format:
        header += ['format_rate']
    if has_length:
        header += ['len_mean', 'len_p50', 'len_p95', 'len_max', 'trunc_rate', 'soft_overlong']

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in datasets:
            e = summary['per_dataset'][d]
            row = [run_name, d, e['correct'], e['total'], f"{e['accuracy']:.4f}"]
            if has_format:
                row += [f"{e.get('format_rate', 0):.4f}" if 'format_rate' in e else '']
            if has_length:
                if 'length' in e:
                    L = e['length']
                    row += [
                        f"{L['mean']:.1f}", L['p50'], L['p95'], L['max'],
                        f"{L['trunc_rate']:.4f}", f"{e['soft_overlong_mean']:.4f}",
                    ]
                else:
                    row += ['', '', '', '', '', '']
            w.writerow(row)
        ov = summary['overall']
        row_micro = [run_name, 'overall_micro', ov['correct'], ov['total'], f"{ov['accuracy_micro']:.4f}"]
        if has_format:
            row_micro += [f"{ov.get('format_rate_micro', 0):.4f}" if 'format_rate_micro' in ov else '']
        if has_length:
            L = ov.get('length', {})
            row_micro += [
                f"{L.get('mean', 0):.1f}", L.get('p50', ''), L.get('p95', ''), L.get('max', ''),
                f"{L.get('trunc_rate', 0):.4f}", f"{ov.get('soft_overlong_mean_micro', 0):.4f}",
            ]
        w.writerow(row_micro)

        row_macro = [run_name, 'overall_macro', '', '', f"{ov['accuracy_macro']:.4f}"]
        if has_format:
            row_macro += [f"{ov.get('format_rate_macro', 0):.4f}" if 'format_rate_macro' in ov else '']
        if has_length:
            row_macro += ['', '', '', '', '', f"{ov.get('soft_overlong_mean_macro', 0):.4f}"]
        w.writerow(row_macro)

    return json_path, csv_path


def print_report(summary: dict, run_name: str):
    order = ['math', 'amc', 'aime']
    known = [d for d in order if d in summary['per_dataset']]
    extras = sorted(d for d in summary['per_dataset'] if d not in order)

    has_format = any('format_rate' in summary['per_dataset'][d] for d in known + extras)
    has_length = any('length' in summary['per_dataset'][d] for d in known + extras)

    print(f"\n=== {run_name} ===")
    hdr = f"{'dataset':<18}{'correct':>10}{'total':>10}{'accuracy':>12}"
    if has_format:
        hdr += f"{'format':>10}"
    if has_length:
        hdr += f"{'len_mean':>10}{'len_p95':>10}{'trunc':>8}{'sol':>8}"
    print(hdr)

    def _row(name, e):
        s = f"{name:<18}{e['correct']:>10}{e['total']:>10}{e['accuracy']*100:>11.2f}%"
        if has_format:
            fr = e.get('format_rate')
            s += f"{fr*100:>9.2f}%" if fr is not None else f"{'-':>10}"
        if has_length:
            L = e.get('length')
            if L is not None:
                s += f"{L['mean']:>10.1f}{L['p95']:>10}{L['trunc_rate']*100:>7.2f}%{e['soft_overlong_mean']:>8.3f}"
            else:
                s += f"{'-':>10}{'-':>10}{'-':>8}{'-':>8}"
        return s

    for d in known + extras:
        print(_row(d, summary['per_dataset'][d]))

    ov = summary['overall']
    micro = {
        'correct': ov['correct'], 'total': ov['total'],
        'accuracy': ov['accuracy_micro'],
        'format_rate': ov.get('format_rate_micro'),
        'length': ov.get('length'),
        'soft_overlong_mean': ov.get('soft_overlong_mean_micro'),
    }
    print(_row('overall (micro)', micro))
    print(f"{'overall (macro)':<18}{'':>10}{'':>10}{ov['accuracy_macro']*100:>11.2f}%",
          end='')
    if has_format and 'format_rate_macro' in ov:
        print(f"{ov['format_rate_macro']*100:>9.2f}%", end='')
    if has_length and 'soft_overlong_mean_macro' in ov:
        print(f"{'':>10}{'':>10}{'':>8}{ov['soft_overlong_mean_macro']:>8.3f}", end='')
    print()



def main():
    parser = argparse.ArgumentParser(description="Generate, grade and summarize math evaluation.")
    parser.add_argument('--model_path', help='Model/checkpoint; required unless --eval-only')
    parser.add_argument('--input_file', '--parquet', default='data/math/valid.all.parquet')
    parser.add_argument('--output_file', '--jsonl', help='Per-sample results (default: OUTPUT_DIR/results.jsonl)')
    parser.add_argument('--output_dir', '--out_dir', help='Summary directory; defaults to results directory')
    parser.add_argument('--eval-only', '--eval_only', action='store_true',
                        help='Summarize existing results without generation or answer regrading')
    parser.add_argument('--tokenizer', help='Tokenizer for length rewards; defaults to model_path')
    parser.add_argument('--decoded', help='Optional generated-text JSONL override')
    parser.add_argument('--run_name')
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('TEMPERATURE', '0')))
    parser.add_argument('--top_p', type=float, default=float(os.environ.get('TOP_P', '1')))
    parser.add_argument('--max_tokens', type=int, default=int(os.environ.get('MAX_TOKENS', '3000')))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('EVAL_SEED', '42')))
    parser.add_argument('--soft_max_length', type=int, default=3000)
    parser.add_argument('--soft_cache_length', type=int, default=200)
    args = parser.parse_args()
    if not args.eval_only and not args.model_path:
        parser.error('--model_path is required for generation')
    if not args.output_file and not args.output_dir:
        parser.error('Provide --output_dir or --output_file')
    if args.soft_max_length <= 0 or args.soft_cache_length <= 0:
        parser.error('Length reward parameters must be positive')
    output_file = args.output_file or os.path.join(args.output_dir, 'results.jsonl')
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(output_file))
    if not os.path.isfile(args.input_file):
        parser.error(f'Missing input dataset: {args.input_file}')
    if args.eval_only:
        if not os.path.isfile(output_file):
            parser.error(f'Missing results: {output_file}')
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        generate_and_grade(args.input_file, output_file, args.model_path,
                           args.temperature, args.top_p, args.max_tokens, args.seed)

    tokenizer = args.tokenizer or args.model_path
    if not tokenizer:
        print('[summary] No tokenizer specified: token lengths and length rewards are omitted.')
    decoded = args.decoded
    if decoded is None:
        sibling = output_file.replace('.jsonl', '') + '.decoded.jsonl'
        if os.path.isfile(sibling):
            decoded = sibling
    run_name = args.run_name or os.path.basename(output_file).replace('.jsonl', '')
    summary = summarize(output_file, args.input_file, decoded_path=decoded,
                        tokenizer_path=tokenizer, soft_max_length=args.soft_max_length,
                        soft_cache_length=args.soft_cache_length)
    print_report(summary, run_name)
    paths = write_outputs(summary, out_dir, run_name)
    print("Wrote: " + ", ".join(paths))


if __name__ == '__main__':
    main()
