#!/usr/bin/env python3
"""ALCE generation and evaluation; scoring stays in eval_alce.py."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def generate(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.eval_set) as f_in:
        records = [json.loads(line) for line in f_in if line.strip()]

    # Data parallel: rank k 处理 index % world_size == k 的样本
    records = [r for i, r in enumerate(records) if i % args.world_size == args.rank]
    out_path = args.output_file
    if args.world_size > 1:
        out_path = f'{args.output_file}.{args.rank}'

    print(f'[gen_{args.task}] rank={args.rank}/{args.world_size} Loading model from {args.model_path}')
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='auto',
        torch_dtype='auto',
        trust_remote_code=True,
    )

    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f'[gen_{args.task}] rank={args.rank} Generating on {len(records)} samples -> {out_path}')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f_out:
        for rec in tqdm(records, desc=f'gen_{args.task}'):
            input_ids = tok.apply_chat_template(
                rec['messages'],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[0, input_ids.size(1):]
            text = tok.decode(gen, skip_special_tokens=True)
            rec = dict(rec)
            rec['output'] = text
            f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'[gen_{args.task}] rank={args.rank} Done. Wrote {out_path}')



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['asqa', 'eli5'], required=True)
    parser.add_argument('--model_path')
    parser.add_argument('--eval_set', help='Defaults to the task test split')
    parser.add_argument('--output_file', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--eval-only', action='store_true', default=os.environ.get('EVAL_ONLY') == '1')
    mode.add_argument('--generate-only', action='store_true',
                      help='Generate without scoring; also supports rank/world_size shards')
    parser.add_argument('--max_new_tokens', type=int,
                        default=int(os.environ['MAX_NEW_TOKENS']) if 'MAX_NEW_TOKENS' in os.environ else None)
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('TEMPERATURE', '1.0')))
    parser.add_argument('--top_p', type=float, default=float(os.environ.get('TOP_P', '0.95')))
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--qa', action='store_true', default=os.environ.get('ENABLE_QA') == '1')
    parser.add_argument('--mauve', action='store_true', default=os.environ.get('ENABLE_MAUVE') == '1')
    args = parser.parse_args(argv)
    if args.generate_only:
        args.eval_only = False
    if not args.eval_only and not args.model_path:
        parser.error('--model_path is required for generation')
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error('Require world_size >= 1 and 0 <= rank < world_size')
    if args.world_size > 1 and not args.generate_only:
        parser.error('Sharding requires --generate-only; merge shards before --eval-only')
    args.eval_set = args.eval_set or f'data/alce/{args.task}_test.jsonl'
    if args.max_new_tokens is None:
        args.max_new_tokens = 400 if args.task == 'asqa' else 500
    return args


def main(argv=None):
    args = parse_args(argv)
    # Keep the old shell entry's repository-relative path semantics.
    os.chdir(ROOT)
    if args.generate_only:
        generate(args)
        return
    if not args.eval_only:
        if not Path(args.eval_set).is_file():
            raise FileNotFoundError(f'Missing dataset: {args.eval_set}')
        # A separate process releases all policy GPU memory before AutoAIS loads.
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), '--generate-only',
            '--task', args.task, '--model_path', args.model_path,
            '--eval_set', args.eval_set, '--output_file', args.output_file,
            '--max_new_tokens', str(args.max_new_tokens),
            '--temperature', str(args.temperature), '--top_p', str(args.top_p),
        ], check=True)
    if not Path(args.output_file).is_file():
        raise FileNotFoundError(f'Missing output: {args.output_file}')
    command = [sys.executable, str(ROOT / 'eval/alce/eval_alce.py'),
               '--f', args.output_file, '--citations']
    if args.task == 'eli5':
        command.append('--claims_nli')
    if args.task == 'asqa' and args.qa:
        command.append('--qa')
    if args.mauve:
        command.append('--mauve')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
