#!/usr/bin/env python3
"""
Offline ALCE evaluation for ms-swift generated outputs.

Mirrors data/ALCE/eval.py, adapted to:
  - Local model paths (read from env vars AUTOAIS_MODEL, QA_MODEL, GPT2_MODEL)
  - Input format from ms-swift infer output OR from the GRPO test-split JSONL
    (eli5_test.jsonl / asqa_test.jsonl) enriched with
    a model-generated `output` field.
  - All complex fields (docs, qa_pairs, annotations, claims) stored as JSON
    strings (as written by convert_alce_to_swift.py) — decoded on the fly.

Input JSON / JSONL format expected
-----------------------------------
Each record must have:
  - output        (str)  : model-generated answer to evaluate
  - question      (str)  : bare question text  (used for MAUVE)
  - answer        (str|JSON-str): gold answer
  - docs          (JSON-str): list of {"title": str, "text": str}
  - dataset_name  (str)  : "asqa" | "eli5"
  - qa_pairs      (JSON-str, ASQA): list of {"short_answers": [...], ...}
  - annotations   (JSON-str, ASQA): list of {"long_answer": str, ...}
  - claims        (JSON-str, ELI5): list of claim strings

The script accepts either:
  A) A JSON file  {"data": [...]} — same structure as ALCE eval.py.
  B) A JSONL file — one record per line.

Usage
-----
  # ELI5 — full evaluation
  python tools/eval_alce.py \\
      --f /path/to/eli5_outputs.json \\
      --citations \\
      --claims_nli \\
      --mauve

  # ASQA — full evaluation
  python tools/eval_alce.py \\
      --f /path/to/asqa_outputs.json \\
      --qa \\
      --citations \\
      --mauve

  # Quick smoke test (no heavy models)
  python tools/eval_alce.py \\
      --f /path/to/outputs.jsonl \\
      --no_rouge

Environment variables
---------------------
  AUTOAIS_MODEL   path to t5_xxl_true_nli_mixture (for --citations and --claims_nli)
  QA_MODEL        path to roberta-large-squad      (for --qa)
  GPT2_MODEL      path to gpt2-large               (for --mauve)
  NLTK_DATA       directory containing punkt data   (optional; regex fallback if absent)
"""

from pathlib import Path as _AMRPPath
_AMRP_ROOT = _AMRPPath(__file__).resolve().parents[2]


import argparse
import collections
import copy
import json
import logging
import os
import re
import string
import sys

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Default model paths (override via env vars)
# ---------------------------------------------------------------------------
AUTOAIS_MODEL = os.environ.get(
    'AUTOAIS_MODEL',
    str(_AMRP_ROOT / './model/t5_xxl_true_nli_mixture'),
)
QA_MODEL = os.environ.get(
    'QA_MODEL',
    str(_AMRP_ROOT / './model/gaotianyu1350/roberta-large-squad'),
)
GPT2_MODEL = os.environ.get(
    'GPT2_MODEL',
    str(_AMRP_ROOT / './model/openai-community/gpt2-large'),
)

_autoais_model = None
_autoais_tokenizer = None


# ---------------------------------------------------------------------------
# Helper utilities (ported from DRBO/alce utils.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def remove_citations(sent: str) -> str:
    return re.sub(r'\[\d+', '', re.sub(r' \[\d+', '', sent)).replace(' |', '').replace(']', '')


def get_max_memory():
    """Return per-GPU max memory dict, reserving headroom for the policy model."""
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_in_GB = int(free_bytes / (1024 ** 3))
        reserve = int(os.environ.get('AUTOAIS_RESERVE_GB', '8'))
        alloc = max(1, free_in_GB - reserve)
        n = torch.cuda.device_count()
        return {i: f'{alloc}GB' for i in range(n)}
    except Exception:
        return None


def sent_tokenize(text: str):
    """NLTK sent_tokenize when available, else regex fallback."""
    nltk_data = os.environ.get('NLTK_DATA')
    if nltk_data and os.path.isdir(nltk_data):
        try:
            import nltk
            if nltk_data not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data)
            from nltk import sent_tokenize as _st
            return _st(text)
        except Exception:
            pass
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def safe_json_loads(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# AutoAIS
# ---------------------------------------------------------------------------

def _load_autoais():
    global _autoais_model, _autoais_tokenizer
    if _autoais_model is not None:
        return
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    logger.info(f'Loading AutoAIS model from {AUTOAIS_MODEL} ...')
    kwargs = dict(torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        # Force single-GPU loading to avoid accelerate multi-GPU device_map
        # issues with T5 cross-attention (CUBLAS_STATUS_EXECUTION_FAILED).
        # T5-XXL (~11B params, ~22GB bfloat16) fits on one 80GB GPU.
        # Override with AUTOAIS_DEVICE env var if needed (e.g. "cuda:1").
        device = os.environ.get('AUTOAIS_DEVICE', 'cuda:0')
        kwargs['device_map'] = {'': device}
    else:
        kwargs['device_map'] = 'cpu'
    _autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, local_files_only=True, **kwargs)
    _autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False, local_files_only=True)
    logger.info('AutoAIS model loaded.')


def _sanitize_text(text: str) -> str:
    """Remove null bytes and control chars that may cause tokenizer to produce invalid token IDs."""
    if not text or not isinstance(text, str):
        return ''
    return ''.join(c for c in text if c == '\n' or c == '\t' or ord(c) >= 32)


def _run_nli_autoais(passage: str, claim: str) -> int:
    global _autoais_model, _autoais_tokenizer
    passage = _sanitize_text(passage)
    claim = _sanitize_text(claim)
    if not passage and not claim:
        return 0
    input_text = f'premise: {passage} hypothesis: {claim}'
    enc = _autoais_tokenizer(
        input_text, return_tensors='pt', truncation=True, max_length=512
    )
    device = next(_autoais_model.parameters()).device
    input_ids = enc.input_ids.to(device)
    # Clamp token IDs to model vocab range to avoid CUDA index out of bounds
    # (T5 tokenizer/model can have vocab size mismatch)
    vocab_size = _autoais_model.config.vocab_size
    input_ids = input_ids.clamp(0, vocab_size - 1)
    pad_token_id = getattr(
        _autoais_tokenizer, 'pad_token_id', None
    ) or _autoais_tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = _autoais_model.generate(
            input_ids, max_new_tokens=10, pad_token_id=pad_token_id
        )
    result = _autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return 1 if result == '1' else 0


def _format_document(doc: dict) -> str:
    if 'sent' in doc:
        return f"Title: {doc['title']}\n{doc['sent']}"
    return f"Title: {doc['title']}\n{doc.get('text', '')}"


# ---------------------------------------------------------------------------
# Metric functions (ported from data/ALCE/eval.py, adapted to ms-swift format)
# ---------------------------------------------------------------------------

def compute_len(data):
    res = sum(len(item['output'].split()) for item in data)
    return res / len(data) if data else 0


def compute_f1(a_gold, a_pred):
    def _get_tokens(s):
        return normalize_answer(s).split() if s else []

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    n_context = normalize_answer(context)
    return any(normalize_answer(sa) in n_context for sa in short_answers)


def compute_rouge(data):
    """ROUGE-Lsum; selects best of two references when annotations present (ASQA)."""
    from rouge_score import rouge_scorer, scoring

    def _calc(hyps, refs1, refs2):
        scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
        agg = scoring.BootstrapAggregator()
        for h, r1, r2 in zip(hyps, refs1, refs2):
            s1 = scorer.score(r1, h)
            s2 = scorer.score(r2, h)
            agg.add_scores(s1 if s1['rougeLsum'].fmeasure >= s2['rougeLsum'].fmeasure else s2)
        if not hyps:
            return 0.0
        return 100 * agg.aggregate()['rougeLsum'].mid.fmeasure

    hyps, refs1, refs2 = [], [], []
    for item in data:
        hyps.append('\n'.join(sent_tokenize(item['output'].lower())))
        annotations = safe_json_loads(item.get('annotations'), None)
        if annotations and isinstance(annotations, list) and len(annotations) >= 2:
            refs1.append('\n'.join(sent_tokenize(annotations[0]['long_answer'].lower())))
            refs2.append('\n'.join(sent_tokenize(annotations[1]['long_answer'].lower())))
        else:
            answer = safe_json_loads(item.get('answer'), '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            ref = '\n'.join(sent_tokenize(str(answer).lower()))
            refs1.append(ref)
            refs2.append(ref)

    return _calc(hyps, refs1, refs2)


def compute_str_em(data):
    """STR-EM and STR-EM-HIT (ASQA only)."""
    acc, hit = [], []
    for item in data:
        qa_pairs = safe_json_loads(item.get('qa_pairs'), [])
        if not qa_pairs:
            acc.append(0.0)
            hit.append(0)
            continue
        loc_acc = [float(exact_presence(qap['short_answers'], item['output'])) for qap in qa_pairs]
        acc.append(float(np.mean(loc_acc)))
        hit.append(int(np.mean(loc_acc) == 1))
    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_qa(data):
    """QA-EM, QA-F1, QA-Hit using a local RoBERTa-large SQuAD model."""
    from transformers import pipeline as hf_pipeline

    logger.info(f'Loading QA model from {QA_MODEL} ...')
    qa_pipe = hf_pipeline('question-answering', model=QA_MODEL, device_map='auto')
    logger.info('QA model loaded.')

    em, f1, bins = [], [], []
    for item in tqdm(data, desc='QA eval'):
        qa_pairs = safe_json_loads(item.get('qa_pairs'), [])
        if not qa_pairs:
            em.append(0.0); f1.append(0.0); bins.append(False)
            continue
        questions = [qp['question'] for qp in qa_pairs]
        ctx = item['output'] if item['output'] else ' '
        results = qa_pipe(question=questions, context=ctx, handle_impossible_answer=True)
        loc_em, loc_f1, cnt = 0, 0, 0
        for idx, res in enumerate(results):
            answers = qa_pairs[idx]['short_answers']
            pred = res['answer']
            loc_em += max(compute_exact(a, pred) for a in answers)
            loc_f1 += max(compute_f1(a, pred) for a in answers)
            cnt += 1
        em.append(loc_em / cnt)
        f1.append(loc_f1 / cnt)
        bins.append(loc_em == cnt)

    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins),
    }


def compute_mauve(data):
    """MAUVE score using GPT-2-large as featurizer (same as DRBO/alce)."""
    import mauve as mauve_lib

    logger.info('Computing MAUVE ...')
    human_data, model_data = [], []
    for item in data:
        q = item.get('question', '')
        answer = safe_json_loads(item.get('answer'), '')
        if isinstance(answer, list):
            answer = answer[0] if answer else ''
        human_data.append(
            ' '.join((q + ' ' + str(answer).strip()).split()[:100]).rstrip(string.punctuation)
        )
        model_data.append(
            ' '.join((q + ' ' + item['output'].strip()).split()[:100]).rstrip(string.punctuation)
        )

    out = mauve_lib.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name=GPT2_MODEL,
    )
    return out.mauve * 100


def compute_claims(data):
    """Claims-NLI (ELI5 correctness). Fraction of gold claims entailed by output."""
    _load_autoais()
    scores = []
    for item in tqdm(data, desc='Claims-NLI'):
        output = remove_citations(item['output'])
        claims = safe_json_loads(item.get('claims'), [])
        if not claims:
            scores.append(0.0)
            continue
        entail = sum(_run_nli_autoais(output, c) for c in claims)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)


def compute_autoais(data, qampari=False, at_most_citations=3):
    """
    AutoAIS citation recall and precision.

    Returns dict with keys 'citation_rec' and 'citation_prec'.
    """
    _load_autoais()
    logger.info('Running AutoAIS ...')

    ais_scores, ais_prec_scores = [], []
    sent_total = sent_mcite = sent_mcite_support = sent_mcite_overcite = 0

    for item in tqdm(data, desc='AutoAIS'):
        docs = safe_json_loads(item.get('docs'), [])
        if qampari:
            sents = [
                item['question'] + ' ' + x.strip()
                for x in item['output'].rstrip().rstrip('.').rstrip(',').split(',')
            ]
        else:
            sents = sent_tokenize(item['output'])
        if not sents:
            continue

        target_sents = [remove_citations(s).strip() for s in sents]
        entail = 0
        entail_prec = 0
        total_citations = 0

        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]
            ref = [int(r[1:]) - 1 for r in re.findall(r'\[\d+', sent)]

            if not ref:
                joint_entail = 0
            elif any(r < 0 or r >= len(docs) for r in ref):
                joint_entail = 0
            else:
                ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(docs[pid]) for pid in ref])
                joint_entail = _run_nli_autoais(joint_passage, target_sent)

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                for psgs_id in ref:
                    passage = _format_document(docs[psgs_id])
                    nli_single = _run_nli_autoais(passage, target_sent)
                    if not nli_single:
                        subset = [pid for pid in ref if pid != psgs_id]
                        passage_sub = '\n'.join([_format_document(docs[pid]) for pid in subset])
                        if _run_nli_autoais(passage_sub, target_sent):
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_prec_scores.append(entail_prec / total_citations if total_citations > 0 else 0.0)

    if sent_mcite > 0 and sent_mcite_support > 0:
        logger.info(
            'Among all sentences, %.2f%% have multiple citations, '
            'among which %.2f%% are supported by the joint set, '
            'among which %.2f%% overcite.' % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support,
            )
        )

    return {
        'citation_rec': 100 * np.mean(ais_scores),
        'citation_prec': 100 * np.mean(ais_prec_scores),
    }


def compute_qampari_f1(data, cot=False):
    prec, rec, rec_top5, f1, f1_top5, num_preds = [], [], [], [], [], []
    for item in data:
        o = item['output']
        if cot and ':' in o:
            o = ':'.join(o.split(':')[1:])
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip('.').rstrip(',').split(',')]
        preds = [p for p in preds if p]
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item.get('answers', [])]
        flat = [x for sub in answers for x in sub]
        _prec = sum(p in flat for p in preds) / len(preds) if preds else 0
        _rec = sum(any(x in preds for x in a) for a in answers) / len(answers) if answers else 0
        _rec5 = min(5, sum(any(x in preds for x in a) for a in answers)) / min(5, len(answers)) if answers else 0
        prec.append(_prec); rec.append(_rec); rec_top5.append(_rec5)
        f1.append(2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0)
        f1_top5.append(2 * _prec * _rec5 / (_prec + _rec5) if (_prec + _rec5) > 0 else 0)
    return {
        'num_preds': float(np.mean(num_preds)),
        'qampari_prec': 100 * np.mean(prec),
        'qampari_rec': 100 * np.mean(rec),
        'qampari_rec_top5': 100 * np.mean(rec_top5),
        'qampari_f1': 100 * np.mean(f1),
        'qampari_f1_top5': 100 * np.mean(f1_top5),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(path: str):
    """Load data from JSON ({"data": [...]}) or JSONL (one record per line)."""
    if path.endswith('.jsonl'):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    else:
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        return obj.get('data', obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Offline ALCE evaluation for ms-swift generated outputs.'
    )
    parser.add_argument('--f', required=True,
                        help='Input file (.json or .jsonl) with `output` field per record.')
    parser.add_argument('--no_rouge', action='store_true', help='Skip ROUGE-Lsum.')
    parser.add_argument('--qa', action='store_true',
                        help='Compute QA-EM/F1/Hit (ASQA; requires local RoBERTa model).')
    parser.add_argument('--mauve', action='store_true',
                        help='Compute MAUVE (requires mauve-text and local GPT-2-large).')
    parser.add_argument('--citations', action='store_true',
                        help='Compute AutoAIS citation_rec / citation_prec.')
    parser.add_argument('--claims_nli', action='store_true',
                        help='Compute Claims-NLI (ELI5 correctness; requires AutoAIS model).')
    parser.add_argument('--at_most_citations', type=int, default=3,
                        help='Max citations per sentence for AutoAIS (default: 3).')
    parser.add_argument('--cot', action='store_true',
                        help='QAMPARI: strip CoT prefix before evaluating.')
    parser.add_argument('--output', default=None,
                        help='Where to write the JSON score file (default: <input>.score).')
    args = parser.parse_args()

    data = load_data(args.f)

    # Detect task type
    qampari = 'qampari' in args.f.lower()
    if qampari:
        args.no_rouge = True
        args.qa = False
        args.mauve = False

    logger.info(f'Loaded {len(data)} records from {args.f}')
    logger.info('Preprocessing outputs ...')

    for i, item in enumerate(data):
        if 'output' not in item or item['output'] is None:
            logger.warning(f'Record {i} has no `output` field; skipping.')
            item['output'] = ''
        raw = item['output'].strip().replace('<|im_end|>', '')
        stripped_think = False
        # Strip <think>...</think> for models that output chain-of-thought (ms-swift inference)
        if '</think>' in raw:
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            stripped_think = True
        elif raw.startswith('<think>'):
            raw = raw[7:].strip()  # Remove opening tag when unclosed
            stripped_think = True
        # Legacy: truncate at first newline only when no think was stripped (ALCE format)
        if not stripped_think and '\n' in raw:
            raw = raw.split('\n')[0]
        data[i]['output'] = raw

    # Separate normalized copy (citations stripped) for non-AutoAIS metrics
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

    result = {}
    result['num_samples'] = len(data)
    result['length'] = compute_len(normalized_data)

    logger.info('Computing STR-EM ...')
    result['str_em'], result['str_hit'] = compute_str_em(normalized_data)

    if qampari:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))

    if not args.no_rouge:
        logger.info('Computing ROUGE-Lsum ...')
        result['rougeLsum'] = compute_rouge(normalized_data)

    if args.qa:
        result.update(compute_qa(normalized_data))

    if args.mauve:
        result['mauve'] = compute_mauve(normalized_data)

    if args.citations:
        result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
        # alce_citation reward = F1 of citation_rec and citation_prec (same as ALCECitationReward)
        rec = result.get('citation_rec', 0) / 100.0
        prec = result.get('citation_prec', 0) / 100.0
        result['citation_f1'] = 100 * (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0

    if args.claims_nli:
        result['claims_nli'] = compute_claims(normalized_data)

    # 与 GRPO 三个 reward 对齐的指标汇总（便于与训练 reward 对比）
    # ASQA 训练用 STR-EM 作为 correctness，ELI5 用 Claims-NLI；评测时未算 claims_nli 则用 str_em
    if not qampari:
        result['grpo_reward_aligned'] = {
            'alce_correctness': result.get('claims_nli') if 'claims_nli' in result else result.get('str_em', 0),  # ASQA: str_em, ELI5: claims_nli (--claims_nli)
            'alce_citation': result.get('citation_f1', 0),      # AutoAIS F1 = 2*rec*prec/(rec+prec)
            'alce_fluency': result['mauve'] if 'mauve' in result else result.get('rougeLsum', 0),
        }

    print(json.dumps(result, indent=2))

    out_path = args.output or (args.f + '.score')
    with open(out_path, 'w') as fout:
        json.dump(result, fout, indent=4)
    logger.info(f'Scores written to {out_path}')


if __name__ == '__main__':
    main()
