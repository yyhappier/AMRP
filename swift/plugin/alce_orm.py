# Copyright (c) Alibaba, Inc. and its affiliates.
"""
ALCE (Automatic LLMs' Citation Evaluation) reward functions for GRPO training.

Supports ASQA (long-form QA) and ELI5 (explanatory QA) datasets.
Three reward dimensions following the ALCE benchmark:
  - Correctness: STR-EM (ASQA) or Claims-NLI (ELI5)
  - Citation:    AutoAIS citation precision/recall F1 (shared)
  - Fluency:     ROUGE-L against gold answer (per-sample proxy for MAUVE)

Data fields expected in each training sample (all complex fields are JSON strings):
  dataset_name : "asqa" | "eli5"
  docs         : JSON string, list of {"title": str, "text": str}
  answer       : JSON string or plain string, the gold answer
  qa_pairs     : JSON string, list of {"short_answers": [str, ...], ...}  (ASQA)
  annotations  : JSON string, list of {"long_answer": str}                (ASQA)
  claims       : JSON string, list of claim strings                       (ELI5)
"""

from pathlib import Path as _AMRPPath
_AMRP_ROOT = _AMRPPath(__file__).resolve().parents[2]


import json
import logging
import os
import re
import string
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (ported from DRBO/alce's utils.py / eval.py)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _remove_citations(sent: str) -> str:
    return re.sub(r'\[\d+', '', re.sub(r' \[\d+', '', sent)).replace(' |', '').replace(']', '')


def _exact_presence(short_answers: List[str], context: str) -> bool:
    n_context = _normalize_answer(context)
    return any(_normalize_answer(sa) in n_context for sa in short_answers)


def _safe_json_loads(value, fallback):
    """Parse a JSON string or return value as-is if already a list/dict."""
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _sent_tokenize(text: str) -> List[str]:
    """Sentence tokenize. Use NLTK only when NLTK_DATA has pre-downloaded data; else regex (no network)."""
    nltk_data = os.environ.get('NLTK_DATA')
    if not nltk_data or not os.path.isdir(nltk_data):
        # No pre-downloaded data: use regex immediately to avoid NLTK urlopen errors.
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    try:
        import nltk
        if nltk_data not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data)
        from nltk import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _format_document(doc: dict) -> str:
    if 'sent' in doc:
        return f"Title: {doc['title']}\n{doc['sent']}"
    return f"Title: {doc['title']}\n{doc.get('text', '')}"


# ---------------------------------------------------------------------------
# AutoAIS NLI runner (shared by Correctness and Citation rewards)
# ---------------------------------------------------------------------------


def _get_max_memory_for_autoais():
    """Per DRBO/alce: limit GPU memory for AutoAIS to avoid OOM/illegal access when sharing with policy.
    Returns dict for device_map, or None if unavailable (e.g. older PyTorch)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info()
        free_in_GB = int(free_bytes / (1024**3))
        reserve_GB = int(os.environ.get('AUTOAIS_RESERVE_GB', '8'))  # Reserve for policy (DRBO uses 6)
        alloc_GB = max(1, free_in_GB - reserve_GB)
        n_gpus = torch.cuda.device_count()
        max_memory = {i: f'{alloc_GB}GB' for i in range(n_gpus)}
        return max_memory
    except Exception:
        return None


class _AutoAISRunner:
    """Singleton wrapper around the T5-based AutoAIS NLI model."""

    _model = None
    _tokenizer = None
    _model_path: Optional[str] = None
    _stream = None  # Dedicated CUDA stream for T5 inference, avoids interference with training stream
    _ds_zero3 = False  # ZeRO-3 下推理需用 disable_deepspeed_zero3 包裹
    # Max input length for NLI to avoid OOM and "sequence longer than max" warnings.
    _nli_max_length = 512

    @classmethod
    def load(cls, model_path: str):
        if cls._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        use_gpu = os.environ.get('USE_AUTOAIS_GPU', '1').strip().lower() in ('1', 'true', 'yes')

        # DeepSpeed ZeRO-3: device_map 不兼容，需 load+to(device)；推理时用 disable_deepspeed_zero3 避免 'weight must be 2-D'
        try:
            from transformers.integrations import is_deepspeed_zero3_enabled
            ds_zero3 = is_deepspeed_zero3_enabled()
        except Exception:
            ds_zero3 = False

        if use_gpu and not ds_zero3:
            current_device = torch.cuda.current_device()
            device_map = {'': current_device}
            load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map)
            logger.info(f'[ALCE] Loading AutoAIS on GPU {current_device}')
        elif use_gpu and ds_zero3:
            load_kwargs = dict(torch_dtype=torch.bfloat16)
            logger.info(f'[ALCE] Loading AutoAIS (ZeRO-3: load to CPU then move to GPU)')
        else:
            device_map = 'cpu'
            load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map)
            logger.info(f'[ALCE] Loading AutoAIS on CPU')

        logger.info(f'[ALCE] Loading AutoAIS model from {model_path} ...')
        cls._model_path = model_path
        cls._ds_zero3 = ds_zero3

        if use_gpu and ds_zero3:
            from swift.utils.utils import disable_deepspeed_zero3
            with disable_deepspeed_zero3():
                cls._model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **load_kwargs)
                target_device = torch.device('cuda', torch.cuda.current_device())
                cls._model = cls._model.to(target_device)
                logger.info(f'[ALCE] AutoAIS moved to GPU {target_device.index}')
        else:
            cls._model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **load_kwargs)
        cls._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        # Ensure tokenizer has proper special tokens
        if cls._tokenizer.pad_token_id is None:
            cls._tokenizer.pad_token_id = cls._tokenizer.eos_token_id
            logger.info(f'[ALCE] Set pad_token_id to eos_token_id: {cls._tokenizer.pad_token_id}')
        if use_gpu:
            dev = next(cls._model.parameters()).device
            if dev.type == 'cuda':
                cls._stream = torch.cuda.Stream(device=dev)
        vocab_size = cls._model.config.vocab_size
        tokenizer_vocab_size = len(cls._tokenizer)

        # Check actual output layer size (lm_head)
        _dev = next(cls._model.parameters()).device
        try:
            lm_head_size = cls._model.lm_head.out_features if hasattr(cls._model, 'lm_head') else None
            logger.info(f'[ALCE] AutoAIS model loaded (device={_dev}, vocab_size={vocab_size}, tokenizer_vocab_size={tokenizer_vocab_size}, lm_head_size={lm_head_size}).')
        except Exception:
            logger.info(f'[ALCE] AutoAIS model loaded (device={_dev}, vocab_size={vocab_size}, tokenizer_vocab_size={tokenizer_vocab_size}).')

        logger.info(f'[ALCE] Special tokens: pad={cls._tokenizer.pad_token_id}, eos={cls._tokenizer.eos_token_id}, bos={getattr(cls._tokenizer, "bos_token_id", None)}')
        if vocab_size != tokenizer_vocab_size:
            logger.warning(f'[ALCE] Vocab size mismatch! model.config.vocab_size={vocab_size} != len(tokenizer)={tokenizer_vocab_size}')

        # DO NOT resize model embeddings - this can cause CUDA indexing issues
        # The T5 model is pre-trained with a specific vocab size and resizing can break it
        if vocab_size != tokenizer_vocab_size:
            logger.warning(f'[ALCE] Vocab size mismatch! model.config.vocab_size={vocab_size} != len(tokenizer)={tokenizer_vocab_size}')
            logger.warning(f'[ALCE] NOT resizing embeddings - will rely on tokenizer to stay within vocab bounds')

    @classmethod
    def _get_device(cls):
        """Device of the first parameter (works with device_map='auto')."""
        return next(cls._model.parameters()).device

    @classmethod
    def _run_in_reward_stream(cls, fn):
        """Run T5 inference on a dedicated stream and sync before return, to avoid CUDA illegal memory access from stream overlap with training."""
        if cls._stream is None:
            return fn()
        import torch
        with torch.cuda.stream(cls._stream):
            out = fn()
        torch.cuda.current_stream().wait_stream(cls._stream)
        return out

    @classmethod
    def run_nli(cls, passage: str, claim: str) -> int:
        """Returns 1 if passage entails claim, else 0."""
        import torch

        def _do_nli():
            # Validate inputs to avoid tokenizer issues
            if not passage or not claim:
                return 0
            if not isinstance(passage, str) or not isinstance(claim, str):
                return 0

            # Follow DRBO/ALCE implementation with truncation to avoid exceeding max length
            input_text = f'premise: {passage} hypothesis: {claim}'
            tok = cls._tokenizer
            # T5 models have max position embeddings of 512, must truncate to avoid indexing errors
            input_ids = tok(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids.to(cls._get_device())

            def _generate():
                with torch.inference_mode():
                    return cls._model.generate(input_ids, max_new_tokens=10)

            try:
                if cls._ds_zero3:
                    from swift.utils.utils import disable_deepspeed_zero3
                    with disable_deepspeed_zero3():
                        outputs = _generate()
                else:
                    outputs = _generate()
            except Exception as e:
                logger.error(f'[AutoAIS] Generation failed: {e}')
                logger.error(f'[AutoAIS] Input shape: {input_ids.shape}')
                return 0

            result = tok.decode(outputs[0], skip_special_tokens=True)
            return 1 if result == '1' else 0

        return cls._run_in_reward_stream(_do_nli)

    @classmethod
    def run_nli_batch(cls, pairs: List[tuple]) -> List[int]:
        """Batch NLI; processes in sub-batches to reduce OOM. pairs = [(passage, claim), ...]."""
        if not pairs:
            return []

        def _do_batch():
            import torch
            batch_size = int(os.environ.get('AUTOAIS_NLI_BATCH_SIZE', '4'))
            results = []
            for i in range(0, len(pairs), batch_size):
                sub = pairs[i : i + batch_size]
                # Validate and filter inputs
                valid_indices = []
                valid_sub = []
                for idx, (p, c) in enumerate(sub):
                    if p and c and isinstance(p, str) and isinstance(c, str):
                        valid_indices.append(idx)
                        valid_sub.append((p, c))

                # Initialize results for this sub-batch
                sub_results = [0] * len(sub)

                if valid_sub:
                    # Follow DRBO/ALCE implementation with truncation to avoid exceeding max length
                    texts = [f'premise: {p} hypothesis: {c}' for p, c in valid_sub]
                    tok = cls._tokenizer
                    # T5 models have max position embeddings of 512, must truncate to avoid indexing errors
                    input_ids = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.to(cls._get_device())

                    def _generate_batch():
                        with torch.inference_mode():
                            return cls._model.generate(input_ids, max_new_tokens=10)

                    try:
                        if cls._ds_zero3:
                            from swift.utils.utils import disable_deepspeed_zero3
                            with disable_deepspeed_zero3():
                                outputs = _generate_batch()
                        else:
                            outputs = _generate_batch()
                        for idx, out in zip(valid_indices, outputs):
                            res = tok.decode(out, skip_special_tokens=True)
                            sub_results[idx] = 1 if res == '1' else 0
                    except Exception as e:
                        logger.error(f'[AutoAIS] Batch generation failed: {e}')
                        logger.error(f'[AutoAIS] Batch size: {len(valid_sub)}, input shape: {input_ids.shape}')
                        # sub_results already initialized to zeros

                results.extend(sub_results)
            return results

        return cls._run_in_reward_stream(_do_batch)


# ---------------------------------------------------------------------------
# Reward 1 – Correctness
# ---------------------------------------------------------------------------

class ALCECorrectnessReward:
    """
    Per-sample correctness reward.

    ASQA  → STR-EM: fraction of qa_pairs whose short_answers appear verbatim in output.
    ELI5  → Claims-NLI: fraction of gold claims entailed by the output (via AutoAIS).

    Returns values in [0, 1].
    """

    def __init__(self, autoais_model_path: Optional[str] = None):
        self.autoais_model_path = (
            autoais_model_path
            or os.environ.get('AUTOAIS_MODEL',
                              str(_AMRP_ROOT / './model/t5_xxl_true_nli_mixture'))
        )

    # ---- ASQA: STR-EM -------------------------------------------------------

    def _compute_str_em(self, completion: str, qa_pairs_raw) -> float:
        output = _remove_citations(completion).strip()
        qa_pairs = _safe_json_loads(qa_pairs_raw, [])
        if not qa_pairs:
            return 0.0
        loc_acc = [float(_exact_presence(qap['short_answers'], output)) for qap in qa_pairs]
        return float(np.mean(loc_acc))

    # ---- ELI5: Claims-NLI ---------------------------------------------------

    def _compute_claims_nli(self, completion: str, claims_raw) -> float:
        try:
            _AutoAISRunner.load(self.autoais_model_path)
        except Exception as e:
            logger.warning(f'[ALCECorrectnessReward] Failed to load AutoAIS model: {e}')
            return 0.0

        output = _remove_citations(completion).strip()
        claims = _safe_json_loads(claims_raw, [])
        if not claims:
            return 0.0

        entail = 0
        for claim in claims:
            try:
                entail += _AutoAISRunner.run_nli(output, claim)
            except Exception as e:
                logger.warning(f'[ALCECorrectnessReward] NLI error for claim: {e}')
                # Skip this claim
                continue

        return entail / len(claims)

    # ---- Public interface ---------------------------------------------------

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        dataset_names = kwargs.get('dataset_name', ['asqa'] * len(completions))
        qa_pairs_list = kwargs.get('qa_pairs', [None] * len(completions))
        claims_list = kwargs.get('claims', [None] * len(completions))

        rewards = []
        for i, (completion, ds_name) in enumerate(zip(completions, dataset_names)):
            try:
                if ds_name == 'asqa':
                    reward = self._compute_str_em(completion, qa_pairs_list[i])
                elif ds_name == 'eli5':
                    reward = self._compute_claims_nli(completion, claims_list[i])
                else:
                    reward = 0.0
            except Exception as e:
                logger.warning(f'[ALCECorrectnessReward] sample {i} error: {e}')
                reward = 0.0
            rewards.append(reward)
        return rewards


# ---------------------------------------------------------------------------
# Reward 2 – Citation Quality (AutoAIS F1)
# ---------------------------------------------------------------------------

class ALCECitationReward:
    """
    Per-sample citation quality reward.

    Uses AutoAIS NLI to verify that each cited document actually supports the
    sentence that cites it (recall), and that no unnecessary documents are cited
    (precision).  Returns the F1 of precision and recall, in [0, 1].

    Works identically for ASQA and ELI5.
    """

    def __init__(self, autoais_model_path: Optional[str] = None, at_most_citations: int = 3):
        self.autoais_model_path = (
            autoais_model_path
            or os.environ.get('AUTOAIS_MODEL',
                              str(_AMRP_ROOT / './model/t5_xxl_true_nli_mixture'))
        )
        self.at_most_citations = at_most_citations

    def _compute_citation_f1(self, completion: str, docs: list) -> float:
        if not docs:
            return 0.0

        try:
            _AutoAISRunner.load(self.autoais_model_path)
        except Exception as e:
            logger.warning(f'[ALCECitationReward] Failed to load AutoAIS model: {e}')
            return 0.0

        sents = _sent_tokenize(completion)
        if not sents:
            return 0.0

        target_sents = [_remove_citations(s).strip() for s in sents]
        entail = 0
        entail_prec = 0
        total_citations = 0

        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]

            # Parse in-text citation indices (1-based → 0-based)
            ref = [int(r[1:]) - 1 for r in re.findall(r'\[\d+', sent)]
            if not ref:
                joint_entail = 0
            elif any(ref_id < 0 or ref_id >= len(docs) for ref_id in ref):
                joint_entail = 0
            else:
                ref = ref[:self.at_most_citations]
                total_citations += len(ref)
                try:
                    joint_passage = '\n'.join([_format_document(docs[pid]) for pid in ref])
                    joint_entail = _AutoAISRunner.run_nli(joint_passage, target_sent)
                except Exception as e:
                    logger.warning(f'[ALCECitationReward] NLI error for sentence {sent_id}: {e}')
                    joint_entail = 0

            entail += joint_entail

            # Precision: check if any cited document is unnecessary
            if joint_entail and len(ref) > 1:
                for psgs_id in ref:
                    try:
                        passage_single = _format_document(docs[psgs_id])
                        nli_single = _AutoAISRunner.run_nli(passage_single, target_sent)
                        if not nli_single:
                            # Try without this document
                            subset = [pid for pid in ref if pid != psgs_id]
                            passage_subset = '\n'.join([_format_document(docs[pid]) for pid in subset])
                            nli_subset = _AutoAISRunner.run_nli(passage_subset, target_sent)
                            if not nli_subset:
                                entail_prec += 1
                            # else: this doc is redundant (overcite), don't credit
                        else:
                            entail_prec += 1
                    except Exception as e:
                        logger.warning(f'[ALCECitationReward] NLI precision error for doc {psgs_id}: {e}')
                        # Skip this citation for precision calculation
                        continue
            else:
                entail_prec += joint_entail

        citation_rec = entail / len(sents)
        citation_prec = entail_prec / total_citations if total_citations > 0 else 0.0

        if (citation_prec + citation_rec) > 0:
            return 2 * citation_prec * citation_rec / (citation_prec + citation_rec)
        return 0.0

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        _AutoAISRunner.load(self.autoais_model_path)
        docs_list = kwargs.get('docs', [None] * len(completions))

        rewards = []
        for i, completion in enumerate(completions):
            try:
                docs = _safe_json_loads(docs_list[i], [])
                reward = self._compute_citation_f1(completion, docs)
            except Exception as e:
                logger.warning(f'[ALCECitationReward] sample {i} error: {e}')
                reward = 0.0
            rewards.append(reward)
        return rewards


# ---------------------------------------------------------------------------
# Reward 3 – Fluency (MAUVE with GPT-2-large, follows DRBO/alce)
# ---------------------------------------------------------------------------

class ALCEFluencyReward:
    """
    Per-batch fluency reward using MAUVE with GPT-2-large as featurizer.

    Follows DRBO/alce compute_mauve exactly:
      - Prepends the question to both model output and gold answer.
      - Truncates to 100 words.
      - Runs mauve.compute_mauve() over the full batch.
      - Returns the same corpus-level MAUVE score (in [0, 1]) for every
        sample in the batch.

    Note: MAUVE is inherently a corpus-level metric.  Per-batch scores
    from small GRPO batches (e.g. 8–16 samples) will be noisier than
    dataset-level MAUVE but still provide a meaningful fluency signal.

    Falls back to per-sample ROUGE-L if MAUVE raises an exception
    (e.g. missing dependency, GPU OOM).

    Requires:
      - pip install mauve-text
      - GPT2_MODEL env var or gpt2_model_path constructor arg pointing
        to a local gpt2-large checkpoint.
    """

    def __init__(self, gpt2_model_path: Optional[str] = None):
        self.gpt2_model_path = (
            gpt2_model_path
            or os.environ.get('GPT2_MODEL',
                              str(_AMRP_ROOT / './model/openai-community/gpt2-large'))
        )
        # 可选：用全训练集 reference 作为 MAUVE 的参考分布（更稳定）
        # 设 ALCE_TRAIN_FILE=/path/to/asqa_train.jsonl 启用
        self._full_train_human_data: Optional[List[str]] = None
        train_file = os.environ.get('ALCE_TRAIN_FILE', '')
        if train_file and os.path.exists(train_file):
            try:
                import json as _json
                items = [_json.loads(l) for l in open(train_file) if l.strip()]
                self._full_train_human_data = []
                for item in items:
                    q = (item.get('question') or '').strip()
                    ans = item.get('answer', '')
                    try: ans = _json.loads(ans)
                    except: pass
                    if isinstance(ans, list): ans = ' '.join(str(a) for a in ans)
                    ans = str(ans)
                    self._full_train_human_data.append(
                        ' '.join((q + ' ' + ans.strip()).split()[:100]).rstrip(string.punctuation)
                    )
                logger.info(f'[ALCEFluencyReward] Loaded {len(self._full_train_human_data)} '
                            f'training references from {train_file}')
            except Exception as e:
                logger.warning(f'[ALCEFluencyReward] Failed to load train file: {e}')

    # ---- helpers ------------------------------------------------------------

    def _rougel(self, hypothesis: str, reference: str) -> float:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
            h = '\n'.join(_sent_tokenize(hypothesis.lower()))
            r = '\n'.join(_sent_tokenize(reference.lower()))
            return scorer.score(r, h)['rougeLsum'].fmeasure
        except Exception:
            return 0.0

    def _get_answer_str(self, answer_raw, annotations_raw) -> Optional[str]:
        """Return a single gold-answer string, preferring long_answer (ASQA)."""
        annotations = _safe_json_loads(annotations_raw, None)
        if annotations and isinstance(annotations, list) and annotations:
            ann = annotations[0]
            if isinstance(ann, dict):
                long_ans = ann.get('long_answer', '')
                if long_ans:
                    return long_ans
        answer = _safe_json_loads(answer_raw, None)
        if answer is None:
            return None
        if isinstance(answer, list):
            return ' '.join(str(a) for a in answer)
        return str(answer)

    # ---- main call ----------------------------------------------------------

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        answers_list     = kwargs.get('answer',      [None] * len(completions))
        annotations_list = kwargs.get('annotations', [None] * len(completions))
        # 'question' field added by convert_alce_to_swift.py
        questions_list   = kwargs.get('question',    ['']   * len(completions))

        clean_completions = [_remove_citations(c).strip() for c in completions]
        references = [
            self._get_answer_str(answers_list[i], annotations_list[i]) or ''
            for i in range(len(completions))
        ]

        # Build human_data / model_data exactly as DRBO/alce compute_mauve
        human_data, model_data = [], []
        for q, ref, hyp in zip(questions_list, references, clean_completions):
            q = (q or '').strip()
            human_data.append(
                ' '.join((q + ' ' + ref.strip()).split()[:100]).rstrip(string.punctuation)
            )
            model_data.append(
                ' '.join((q + ' ' + hyp.strip()).split()[:100]).rstrip(string.punctuation)
            )

        try:
            if os.environ.get('USE_MAUVE', '1') == '0':
                raise ValueError('USE_MAUVE=0: using ROUGE-L')
            import mauve as mauve_lib
            import torch
            import torch.distributed as dist

            # ── 0. 一次性 patch: 用 GatheredParameters 恢复 ZeRO-3 展平的参数
            import sys
            _cm_mod = sys.modules.get('mauve.compute_mauve')
            if _cm_mod is not None and not getattr(_cm_mod, '_zero3_patched', False):
                _orig_featurize = _cm_mod.featurize_tokens_from_model
                def _gathered_featurize(model, *args, **kwargs):
                    import deepspeed
                    params = list(model.parameters())
                    with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
                        result = _orig_featurize(model, *args, **kwargs)
                    # bfloat16 → float32 for numpy compatibility
                    return result.float()
                _cm_mod.featurize_tokens_from_model = _gathered_featurize
                _cm_mod._zero3_patched = True

            # ── 1. 聚合所有 rank 的文本（与 DRBO 的全 batch 计算对齐）
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                all_human = [None] * world_size
                all_model = [None] * world_size
                dist.all_gather_object(all_human, human_data)
                dist.all_gather_object(all_model, model_data)
                gathered_human = [x for sub in all_human for x in sub]
                gathered_model = [x for sub in all_model for x in sub]
            else:
                gathered_human = human_data
                gathered_model = model_data

            if len(gathered_human) < 10:
                raise ValueError(f'Too few samples ({len(gathered_human)}) for MAUVE')

            # ── 2. 计算 MAUVE
            #    若设了 ALCE_TRAIN_FILE，用全训练集作为参考分布（更稳定）
            #    否则用当前 batch 的 gathered_human（与 DRBO 一致）
            ref_texts = self._full_train_human_data if self._full_train_human_data else gathered_human
            out = mauve_lib.compute_mauve(
                p_text=ref_texts,
                q_text=gathered_model,
                device_id=0,
                max_text_length=512,
                verbose=False,
                batch_size=8,
                featurize_model_name=self.gpt2_model_path,
            )
            score = float(out.mauve)  # [0, 1]，与 correctness/citation 量纲一致
            ref_src = f'full_train({len(ref_texts)})' if self._full_train_human_data else f'batch({len(gathered_human)})'
            logger.info(f'[ALCEFluencyReward] MAUVE = {score:.4f} '
                        f'(ref={ref_src}, model={len(gathered_model)} samples, '
                        f'world_size={dist.get_world_size() if dist.is_initialized() else 1})')
            return [score] * len(completions)
        except Exception as e:
            msg = f'[ALCEFluencyReward] MAUVE failed ({e}), falling back to ROUGE-L'
            if isinstance(e, (ValueError, NotImplementedError)):
                logger.debug(msg)
            else:
                import traceback as _tb
                logger.warning(msg + '\n' + _tb.format_exc())
            return [
                self._rougel(hyp, ref) if ref else 0.0
                for hyp, ref in zip(clean_completions, references)
            ]
