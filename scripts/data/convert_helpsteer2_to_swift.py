#!/usr/bin/env python3
"""
Convert HelpSteer2 dataset to ms-swift GRPO format.

Input  : HelpSteer2 JSONL(.gz) — fields: prompt, response, helpfulness,
         correctness, coherence, complexity, verbosity
Output : JSONL where each row has:
           messages   : [{"role": "user", "content": <prompt>}]
           helpfulness: int 0-4   (reward signal for hs2_helpfulness ORM)
           correctness: int 0-4   (reward signal for hs2_correctness ORM)
           coherence  : int 0-4   (reward signal for hs2_coherence  ORM)

Usage:
  python3 scripts/data/convert_helpsteer2_to_swift.py \\
      --input  /path/to/HelpSteer2/train.jsonl.gz \\
      --output /path/to/train_v2.jsonl

  python3 scripts/data/convert_helpsteer2_to_swift.py \\
      --input  /path/to/HelpSteer2/validation.jsonl.gz \\
      --output /path/to/helpsteer2_grpo_val.jsonl
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def _open(path: str):
    p = Path(path)
    if p.suffix == '.gz':
        return gzip.open(p, 'rt', encoding='utf-8')
    return open(p, 'r', encoding='utf-8')


def parse_prompt(prompt: str) -> list:
    """
    Parse HelpSteer2 prompt into a messages list.

    Single-turn:  "some question"
      → [{"role": "user", "content": "some question"}]

    Multi-turn:   "{user1}\n<extra_id_1>Assistant\n{asst1}\n<extra_id_1>User\n{user2}"
      → [{"role": "user", "content": "{user1}"},
         {"role": "assistant", "content": "{asst1}"},
         {"role": "user", "content": "{user2}"}]

    The response field is always the final assistant turn and is NOT included
    here; callers append it separately as needed.
    """
    if '<extra_id_1>' not in prompt:
        return [{'role': 'user', 'content': prompt.strip()}]

    # Split on <extra_id_1> boundaries; first segment is always the first user turn
    parts = prompt.split('<extra_id_1>')
    messages = [{'role': 'user', 'content': parts[0].strip()}]
    for part in parts[1:]:
        # Each part starts with "Assistant\n..." or "User\n..."
        if part.startswith('Assistant\n'):
            messages.append({'role': 'assistant', 'content': part[len('Assistant\n'):].strip()})
        elif part.startswith('User\n'):
            messages.append({'role': 'user', 'content': part[len('User\n'):].strip()})
        # ignore unexpected segments
    return messages


def convert(input_path: str, output_path: str, min_score: int = 0) -> None:
    kept = skipped = 0
    with _open(input_path) as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            helpfulness = row.get('helpfulness')
            correctness = row.get('correctness')
            coherence = row.get('coherence')

            if any(v is None for v in (helpfulness, correctness, coherence)):
                skipped += 1
                continue

            if min_score > 0 and (helpfulness + correctness + coherence) < min_score:
                skipped += 1
                continue

            out = {
                'messages': parse_prompt(row['prompt']),
                'helpfulness': int(helpfulness),
                'correctness': int(correctness),
                'coherence': int(coherence),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            kept += 1

    print(f'Wrote {kept} samples to {output_path}  (skipped {skipped})', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Convert HelpSteer2 → ms-swift GRPO format')
    parser.add_argument('--input', required=True, help='Path to HelpSteer2 JSONL or JSONL.gz')
    parser.add_argument('--output', required=True, help='Output JSONL path')
    parser.add_argument(
        '--min_score',
        type=int,
        default=0,
        help='Keep only samples where helpfulness+correctness+coherence >= this value (0 = keep all)')
    args = parser.parse_args()
    convert(args.input, args.output, args.min_score)


if __name__ == '__main__':
    main()
