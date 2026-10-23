#!/usr/bin/env python3
"""
Convert ALCE evaluation JSON files into ms-swift GRPO-compatible JSONL.

Supports both ASQA and ELI5 dataset types.

Usage:
  # ASQA (defaults aligned with DRBO/alce: ndoc=3, shot=1)
  python scripts/data/convert_alce_to_swift.py \
      --eval_file /path/to/alce/data/asqa_eval_gtr_top100_reranked_oracle.json \
      --prompt_file /path/to/alce/prompts/asqa_default.json \
      --dataset_name asqa \
      --output_file /path/to/output/asqa_train.jsonl

  # ELI5
  python scripts/data/convert_alce_to_swift.py \
      --eval_file /path/to/alce/data/eli5_eval_bm25_top100_reranked_oracle.json \
      --prompt_file /path/to/alce/prompts/eli5_default.json \
      --dataset_name eli5 \
      --output_file /path/to/output/eli5_train.jsonl

Output JSONL format per line:
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "<instruction + few-shot demos + question + docs>"}
  ],
  "dataset_name": "asqa" | "eli5",
  "question":     "...",                                       // plain string, used by ALCEFluencyReward (MAUVE)
  "docs":         "[{\"title\": ..., \"text\": ...}, ...]",   // JSON string
  "answer":       "...",                                       // string
  "qa_pairs":     "[{\"short_answers\": [...], ...}]",        // JSON string (ASQA only)
  "annotations":  "[{\"long_answer\": \"...\"}, ...]",        // JSON string (ASQA only)
  "claims":       "[\"...\", ...]"                             // JSON string (ELI5 only)
}

The 'messages' field contains the full ALCE-style prompt so that the LLM
knows it must produce an answer that cites documents using [1][2][3] notation.
All other fields are carried through as JSON strings for use by the reward
functions (ALCECorrectnessReward, ALCECitationReward, ALCEFluencyReward).
"""

import argparse
import json
import os
import re
import numpy as np
import string
import sys


SYSTEM_PROMPT = (
    'You are an intelligent assistant to help me answer questions based on '
    'context. Always cite documents using [1][2][3] notation.'
)


# ---------------------------------------------------------------------------
# Prompt construction (mirrors DRBO/alce utils.py)
# ---------------------------------------------------------------------------

def make_doc_prompt(doc: dict, doc_id: int, doc_prompt_tpl: str,
                    use_shorter: str = None) -> str:
    text = doc.get('text', '')
    if use_shorter and use_shorter in doc:
        text = doc[use_shorter]
    return (doc_prompt_tpl
            .replace('{ID}', str(doc_id + 1))
            .replace('{T}', doc.get('title', ''))
            .replace('{P}', text))


def make_demo(item: dict, prompt_tpl: str, ndoc: int,
              doc_prompt_tpl: str, instruction: str,
              use_shorter: str = None, test: bool = False) -> str:
    prompt = prompt_tpl.replace('{INST}', instruction).replace('{Q}', item['question'])
    if '{D}' in prompt:
        if ndoc == 0:
            prompt = prompt.replace('{D}\n', '')
        else:
            docs = item.get('docs', [])[:ndoc]
            doc_text = ''.join(
                make_doc_prompt(doc, i, doc_prompt_tpl, use_shorter)
                for i, doc in enumerate(docs)
            )
            prompt = prompt.replace('{D}', doc_text)

    if not test:
        answer = item.get('answer', '')
        if isinstance(answer, list):
            answer = '\n' + '\n'.join(answer)
        prompt = prompt.replace('{A}', '').rstrip() + answer
    else:
        prompt = prompt.replace('{A}', '').rstrip()

    return prompt


def build_head_prompt(prompt_data: dict, shot: int, ndoc: int,
                      use_shorter: str = None, seed: int = 2024) -> str:
    np.random.seed(seed)
    demos = prompt_data.get('demos', [])
    chosen = np.random.choice(len(demos), min(shot, len(demos)), replace=False).tolist()

    head = ''
    for idx, demo_id in enumerate(chosen):
        item = demos[demo_id]
        instruction = prompt_data['instruction'] if idx == 0 else ''
        head += make_demo(
            item,
            prompt_tpl=prompt_data['demo_prompt'],
            ndoc=ndoc,
            doc_prompt_tpl=prompt_data['doc_prompt'],
            instruction=instruction,
            use_shorter=use_shorter,
        )
        head += prompt_data.get('demo_sep', '\n\n\n')
    return head


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(eval_file: str, prompt_file: str, dataset_name: str,
            ndoc: int, shot: int, use_shorter: str, seed: int,
            output_file: str, train_ratio: float):
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    with open(prompt_file, 'r') as f:
        prompt_data = json.load(f)

    head_prompt = build_head_prompt(prompt_data, shot=shot, ndoc=ndoc,
                                    use_shorter=use_shorter, seed=seed)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # 80/20 split: train = 前 80%，test = 剩余 20%，不丢弃样本（DRBO 用 eval_data[-int(n*0.2):] 做 test 时，当 n=948 会漏掉下标 758 的样本）
    n_total = len(eval_data)
    n_train = int(n_total * train_ratio)

    train_data = eval_data[:n_train]
    test_data = eval_data[n_train:]

    def _make_record(item: dict) -> dict:
        # Build the user-facing prompt (no answer, test=True)
        user_content = head_prompt + make_demo(
            item,
            prompt_tpl=prompt_data['demo_prompt'],
            ndoc=ndoc,
            doc_prompt_tpl=prompt_data['doc_prompt'],
            instruction='',
            use_shorter=use_shorter,
            test=True,
        )

        # Truncate docs to ndoc for reward computation
        docs = item.get('docs', [])[:ndoc]

        record = {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': user_content},
            ],
            'dataset_name': dataset_name,
            'question':     item.get('question', ''),
            'docs':         json.dumps(docs, ensure_ascii=False),
            'answer':       json.dumps(item.get('answer', ''), ensure_ascii=False),
        }

        if dataset_name == 'asqa':
            record['qa_pairs']    = json.dumps(item.get('qa_pairs', []),    ensure_ascii=False)
            record['annotations'] = json.dumps(item.get('annotations', []), ensure_ascii=False)

        elif dataset_name == 'eli5':
            record['claims'] = json.dumps(item.get('claims', []), ensure_ascii=False)

        return record

    # Write train split
    train_out = output_file
    test_out  = output_file.replace('.jsonl', '_test.jsonl')

    with open(train_out, 'w', encoding='utf-8') as fw:
        for item in train_data:
            fw.write(json.dumps(_make_record(item), ensure_ascii=False) + '\n')

    with open(test_out, 'w', encoding='utf-8') as fw:
        for item in test_data:
            fw.write(json.dumps(_make_record(item), ensure_ascii=False) + '\n')

    print(f'[convert_alce] dataset  : {dataset_name}')
    print(f'[convert_alce] total    : {n_total}  (train {n_train}, test {n_total - n_train})')
    print(f'[convert_alce] ndoc     : {ndoc}')
    print(f'[convert_alce] shot     : {shot}')
    print(f'[convert_alce] train -> : {train_out}')
    print(f'[convert_alce] test  -> : {test_out}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Convert ALCE JSON to ms-swift GRPO JSONL')
    parser.add_argument('--eval_file',    required=True,
                        help='Path to ALCE eval JSON (e.g. asqa_eval_gtr_top100_reranked_oracle.json)')
    parser.add_argument('--prompt_file',  required=True,
                        help='Path to ALCE prompt JSON (e.g. asqa_default.json)')
    parser.add_argument('--dataset_name', required=True, choices=['asqa', 'eli5'],
                        help='Dataset type')
    parser.add_argument('--output_file',  required=True,
                        help='Output JSONL path for the training split')
    parser.add_argument('--ndoc',         type=int, default=3,
                        help='Number of documents per sample (default: 3, align DRBO/alce)')
    parser.add_argument('--shot',         type=int, default=1,
                        help='Few-shot demos to prepend (default: 1, align DRBO/alce)')
    parser.add_argument('--use_shorter',  type=str, default=None,
                        choices=[None, 'summary', 'extraction'],
                        help='Use summary/extraction field instead of full text')
    parser.add_argument('--seed',         type=int, default=2024,
                        help='Random seed for demo sampling (default: 2024, align DRBO/alce)')
    parser.add_argument('--train_ratio',  type=float, default=0.8,
                        help='Fraction of data to use for training (default: 0.8)')
    args = parser.parse_args()

    convert(
        eval_file=args.eval_file,
        prompt_file=args.prompt_file,
        dataset_name=args.dataset_name,
        ndoc=args.ndoc,
        shot=args.shot,
        use_shorter=args.use_shorter,
        seed=args.seed,
        output_file=args.output_file,
        train_ratio=args.train_ratio,
    )


if __name__ == '__main__':
    main()
