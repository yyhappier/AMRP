# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate.utils import gather
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer
from trl import PPOTrainer as HFPPOTrainer

from swift.llm import RowPreprocessor
from swift.plugin import orms
from swift.utils import get_logger, patch_getattr
from ..mixin import SwiftMixin
from .utils import get_gather_if_zero3_context, set_expandable_segments

ppo_trainer_init = HFPPOTrainer.__init__
del HFPPOTrainer.__init__
logger = get_logger()


class _RuleRewardBackbone(nn.Module):

    def __init__(self, anchor: nn.Parameter):
        super().__init__()
        self.anchor = anchor

    def forward(self, input_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden = self.anchor.reshape(1, 1, -1).expand(batch_size, seq_len, -1)
        return SimpleNamespace(hidden_states=(hidden, ))


class _RuleRewardModel(nn.Module):
    base_model_prefix = 'backbone'

    def __init__(self, hidden_size: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.backbone = _RuleRewardBackbone(self.anchor)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden_size)


class PPOTrainer(SwiftMixin, HFPPOTrainer):

    @staticmethod
    @contextmanager
    def _patch_dataloader(collate_fn):
        __init__ = DataLoader.__init__

        def __new_init__(self, *args, **kwargs):
            kwargs['collate_fn'] = collate_fn
            __init__(self, *args, **kwargs)

        DataLoader.__init__ = __new_init__
        try:
            yield
        finally:
            DataLoader.__init__ = __init__

    def __init__(self, model: PreTrainedModel, ref_model: PreTrainedModel, *_args, **kwargs):
        rule_reward_funcs = kwargs.pop('reward_funcs', None)
        rule_reward_weights = kwargs.pop('reward_weights', None)
        rule_reward_func_kwargs = kwargs.pop('reward_func_kwargs', None) or {}
        self._ppo_rule_reward_enabled = bool(rule_reward_funcs)
        self._ppo_rule_reward_funcs = []
        self._ppo_rule_reward_func_names = []
        self._ppo_rule_reward_weights = None
        self._ppo_rule_reward_weight_values = rule_reward_weights
        self._ppo_rule_reward_func_kwargs = rule_reward_func_kwargs
        self._ppo_rule_reward_kwargs_by_prompt = {}
        self._ppo_rule_reward_pending_metrics = {}
        self._ppo_rule_reward_in_generate = False
        self._swift_no_eval_dataset = self._is_empty_dataset(kwargs.get('eval_dataset'))
        if self._ppo_rule_reward_enabled:
            if kwargs.get('value_model') is None:
                raise ValueError('PPO with rule-based `reward_funcs` requires `value_model`; pass `--reward_model` '
                                 'to initialize the PPO value model.')
            if kwargs.get('reward_model') is None:
                kwargs['reward_model'] = _RuleRewardModel(self._get_hidden_size(kwargs['value_model'], model))
        super().__init__(model, *_args, **{k: v for k, v in kwargs.items() if k not in {'reward_model', 'value_model'}})
        if self._swift_no_eval_dataset:
            kwargs['eval_dataset'] = []
        with self._patch_dataloader(kwargs['data_collator']):
            new_kwargs = {
                k: v
                for k, v in kwargs.items() if k in [
                    'train_dataset',
                    'data_collator',
                    'reward_model',
                    'value_model',
                    'eval_dataset',
                    'callbacks',
                ]
            }
            parameters = inspect.signature(ppo_trainer_init).parameters
            if 'config' in parameters:
                new_kwargs['config'] = kwargs['args']
            else:
                new_kwargs['args'] = kwargs['args']
            if 'processing_class' in parameters:
                new_kwargs['processing_class'] = self.tokenizer
            else:
                new_kwargs['tokenizer'] = self.tokenizer
            ppo_trainer_init(self, model=model, ref_model=ref_model, **new_kwargs)
        if self._ppo_rule_reward_enabled:
            self._prepare_rule_rewards(rule_reward_funcs)
            self._prepare_rule_reward_weighting()
            self._build_rule_reward_index()
        unwrap_model = self.accelerator.unwrap_model(self.model)
        patch_getattr(unwrap_model.__class__, 'policy')

        self.use_vllm = bool(getattr(self.args, 'use_vllm', False))
        self.vllm_engine = None
        self._vllm_last_loaded_step = -1
        if self.use_vllm:
            self._init_vllm_engine()

    def create_loss_and_metric(self, args):
        return {}

    # -------------------- vLLM rollout backend (colocate) --------------------
    def _init_vllm_engine(self):
        """Build a colocate vLLM engine that shares the GPU with training via sleep_level.

        Mirrors ``RolloutTrainerMixin._prepare_vllm_engine`` but is self-contained: PPO already
        holds tokenized ``queries`` tensors, so we drive vLLM with ``prompt_token_ids`` directly
        instead of the message/template pipeline.
        """
        from copy import copy
        from swift.tuners import Swift
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        args = self.args
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        if self.vllm_mode != 'colocate':
            raise ValueError(f'PPO vLLM backend only supports colocate mode, got vllm_mode={self.vllm_mode!r}.')
        if args.vllm_tensor_parallel_size != 1:
            raise ValueError('PPO vLLM backend currently supports only vllm_tensor_parallel_size=1.')

        policy_model = self.policy_model  # original passed-in model: has model_dir/model_info/model_meta
        world_size = self.accelerator.num_processes
        # one request per prompt per micro-batch; over-provision by rollout batch size
        max_num_seqs = max(int(getattr(args, 'local_rollout_forward_batch_size', 1) or 1),
                           int(getattr(args, 'per_device_train_batch_size', 1) or 1))
        # IMPORTANT: give vLLM its own template copy. The engine calls template.set_mode('vllm'),
        # which would otherwise corrupt the shared training template's mode and break the PPO
        # data collator (UnboundLocalError on 'res').
        vllm_template = copy(self.template)
        vllm_template.padding_free = False

        with Swift.grpo_context(policy_model, self.template.processor):
            set_expandable_segments(False)
            self.vllm_engine = GRPOVllmEngine(
                policy_model.model_dir,
                policy_model.model_info.torch_dtype,
                model_type=policy_model.model_meta.model_type,
                use_async_engine=False,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                enable_prefix_caching=args.vllm_enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                enforce_eager=args.vllm_enforce_eager,
                limit_mm_per_prompt=args.vllm_limit_mm_per_prompt,
                enable_sleep_mode=args.sleep_level > 0,
                max_model_len=args.vllm_max_model_len,
                seed=self.accelerator.process_index,
                disable_cascade_attn=args.vllm_disable_cascade_attn,
                load_format='dummy',  # weights are synced from the policy after init
                mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
                template=vllm_template,
                distributed_executor_backend='external_launcher' if world_size > 1 else None,
                engine_kwargs=args.vllm_engine_kwargs,
            )
            set_expandable_segments(True)
        if args.sleep_level > 0:
            self.vllm_engine.engine.sleep(args.sleep_level)
        logger.info(f'PPO vLLM colocate engine ready (sleep_level={args.sleep_level}, '
                    f'gpu_memory_utilization={self.vllm_gpu_memory_utilization}, '
                    f'max_model_len={args.vllm_max_model_len}).')

    def _move_policy_to_vllm(self):
        """Sync current policy weights from the (possibly ZeRO-3 sharded) trainer to vLLM.

        The training model is a ``PolicyAndValueWrapper``; policy params carry a ``policy.`` prefix.
        vLLM only needs the bare policy weights, so we strip that prefix.
        """
        gather_if_zero3 = get_gather_if_zero3_context(self)
        model = self.accelerator.unwrap_model(self.model)
        policy_params = [p for n, p in model.named_parameters() if n.startswith('policy.')]
        with gather_if_zero3(policy_params):
            state_dict = {
                name[len('policy.'):]: param
                for name, param in model.state_dict().items() if name.startswith('policy.')
            }
            llm_model = self.vllm_engine.inner_model
            llm_model.load_weights(state_dict.items())
            del state_dict
        self.vllm_engine.engine.reset_prefix_cache()

    def _vllm_wake_up(self):
        if self.args.sleep_level > 0:
            self.vllm_engine.engine.wake_up()

    def _vllm_sleep(self):
        if self.args.sleep_level > 0:
            self.vllm_engine.engine.sleep(self.args.sleep_level)

    def _vllm_generate(self, queries: torch.Tensor, generation_config) -> torch.Tensor:
        """Generate responses for ``queries`` [B, ctx] with vLLM.

        Returns ``query_responses`` [B, ctx + max_resp] laid out exactly like trl's
        ``batch_generation``: original queries (left padding preserved) concatenated with the
        generated tokens, right-padded to the longest response with ``pad_token_id``.
        """
        from vllm import SamplingParams
        pad_token_id = self.processing_class.pad_token_id
        device = queries.device
        max_new_tokens = int(generation_config.max_new_tokens)

        prompt_token_ids = []
        for row in queries.tolist():
            start = 0
            while start < len(row) and row[start] == pad_token_id:
                start += 1
            prompt_token_ids.append([int(t) for t in row[start:]])

        # Mirror trl's generation_config exactly (top_k=0.0 / top_p=1.0 => no truncation,
        # pure temperature sampling) so the vLLM rollout matches the native PPO distribution.
        # vLLM: top_k=0 means disabled; top_p must be in (0, 1].
        gc_temperature = float(getattr(generation_config, 'temperature', None) or (self.args.temperature + 1e-7))
        gc_top_p = getattr(generation_config, 'top_p', None)
        gc_top_k = getattr(generation_config, 'top_k', None)
        top_p = float(gc_top_p) if gc_top_p not in (None, 0, 0.0) else 1.0
        top_k = int(gc_top_k) if gc_top_k not in (None, ) else 0
        if top_k < 0:
            top_k = 0
        sampling_params = SamplingParams(
            n=1,
            temperature=gc_temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            detokenize=False,
        )

        request_id_list = []
        for prompt_ids in prompt_token_ids:
            request_id = f'ppo-{self.vllm_engine._request_count}'
            self.vllm_engine._request_count += 1
            request_id_list.append(request_id)
            self.vllm_engine.engine.add_request(request_id, {'prompt_token_ids': prompt_ids}, sampling_params)

        outputs = {}
        while self.vllm_engine.engine.has_unfinished_requests():
            for out in self.vllm_engine.engine.step():
                if out.finished:
                    outputs[out.request_id] = out
        responses = [list(outputs[rid].outputs[0].token_ids) for rid in request_id_list]

        max_resp = max(max((len(r) for r in responses), default=0), 1)
        context_length = queries.shape[1]
        query_responses = torch.full((queries.shape[0], context_length + max_resp),
                                     pad_token_id, dtype=queries.dtype, device=device)
        query_responses[:, :context_length] = queries
        for i, resp in enumerate(responses):
            if resp:
                query_responses[i, context_length:context_length + len(resp)] = torch.tensor(
                    resp, dtype=queries.dtype, device=device)
        return query_responses

    def _recompute_generation_logits(self, policy_model, query_responses: torch.Tensor,
                                     context_length: int) -> torch.Tensor:
        """Recompute behavior-policy logits over the vLLM-generated tokens.

        trl uses the generation-time (temperature-warped) scores as the behavior logits (the PPO
        ratio denominator). Since vLLM returns only token ids, we recover numerically consistent
        logits with one policy forward pass and apply the same temperature scaling trl applies to
        the reference logits (ppo_trainer.py: ``ref_logits /= args.temperature``).
        """
        pad_token_id = self.processing_class.pad_token_id
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        output = policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output.logits[:, context_length - 1:-1]
        logits = logits / (self.args.temperature + 1e-7)
        return logits

    def _patched_batch_generation(self, model, queries, local_rollout_forward_batch_size, pad_token_id,
                                  generation_config):
        """Drop-in replacement for trl's ``batch_generation`` using the vLLM backend."""
        if self._vllm_last_loaded_step != self.state.global_step:
            self._vllm_wake_up()
            self._move_policy_to_vllm()
            self._vllm_last_loaded_step = self.state.global_step

        context_length = queries.shape[1]
        query_responses = self._vllm_generate(queries, generation_config)

        # Under ZeRO-3 the policy params were re-sharded to 1-D when _move_policy_to_vllm's
        # GatheredParameters context exited (and trl removed the auto-gather hooks for this
        # generation window), so explicitly re-gather the full policy weights once around the
        # whole recompute loop rather than per micro-batch.
        gather_if_zero3 = get_gather_if_zero3_context(self)
        logitss = []
        with gather_if_zero3(list(model.parameters())):
            for i in range(0, query_responses.shape[0], local_rollout_forward_batch_size):
                qr = query_responses[i:i + local_rollout_forward_batch_size]
                logitss.append(self._recompute_generation_logits(model, qr, context_length))
        logitss = torch.cat(logitss, dim=0)

        # vLLM is no longer needed until the next step's rollout. Put it to sleep now so its GPU
        # (~42 GiB weights + KV cache) is freed for the memory-heavy PPO forward/backward update.
        # The next _patched_batch_generation call re-wakes it (gated by _vllm_last_loaded_step).
        self._vllm_sleep()
        self._vllm_last_loaded_step = -1
        torch.cuda.empty_cache()
        return query_responses, logitss

    @contextmanager
    def _patch_vllm_generation(self):
        """Swap trl's ``batch_generation`` for the vLLM-backed one during ``train()``."""
        if not self.use_vllm:
            yield
            return
        train_globals = HFPPOTrainer.train.__globals__
        original = train_globals['batch_generation']

        def patched_batch_generation(model, queries, local_rollout_forward_batch_size, pad_token_id,
                                     generation_config):
            return self._patched_batch_generation(model, queries, local_rollout_forward_batch_size,
                                                   pad_token_id, generation_config)

        train_globals['batch_generation'] = patched_batch_generation
        try:
            yield
        finally:
            train_globals['batch_generation'] = original
            # After training, _patched_batch_generation has already left vLLM asleep. Leaving it
            # asleep on process exit causes vLLM's CuMemAllocator destructor to try to free the
            # discarded (slept) memory pool -> "Trying to free a pointer not allocated here"
            # SIGSEGV during teardown. Wake it back up so the allocator holds a valid pool at exit.
            self._vllm_wake_up()

    @staticmethod
    def _is_empty_dataset(dataset) -> bool:
        if dataset is None:
            return True
        if isinstance(dataset, dict):
            return len(dataset) == 0 or all(PPOTrainer._is_empty_dataset(ds) for ds in dataset.values())
        try:
            return len(dataset) == 0
        except TypeError:
            return False

    @staticmethod
    def _get_hidden_size(*models) -> int:
        for model in models:
            config = getattr(model, 'config', None)
            hidden_size = getattr(config, 'hidden_size', None)
            if hidden_size is not None:
                return hidden_size
            hidden_sizes = getattr(config, 'hidden_sizes', None)
            if hidden_sizes:
                return max(hidden_sizes)
        return 1

    @staticmethod
    def _normalize_token_ids(token_ids, pad_token_id) -> tuple:
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if pad_token_id is None:
            return tuple(int(token) for token in token_ids)
        token_ids = [int(token) for token in token_ids]
        start, end = 0, len(token_ids)
        while start < end and token_ids[start] == pad_token_id:
            start += 1
        while end > start and token_ids[end - 1] == pad_token_id:
            end -= 1
        return tuple(token_ids[start:end])

    @staticmethod
    def _first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
        row_len = bools.size(-1)
        zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
        return torch.min(zero_or_index, dim=-1).values

    @staticmethod
    def _row_extra_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
        extra_kwargs = row.get('_extra_kwargs') or {}
        extra_kwargs = dict(extra_kwargs) if isinstance(extra_kwargs, dict) else {}
        if 'solution' in row and 'solution' not in extra_kwargs:
            extra_kwargs['solution'] = row['solution']
        return extra_kwargs

    @staticmethod
    def _get_reward_call_parameters(reward_func):
        if inspect.isfunction(reward_func) or inspect.ismethod(reward_func):
            return inspect.signature(reward_func).parameters
        return inspect.signature(reward_func.__call__).parameters

    def _iter_rule_reward_datasets(self):
        datasets = [getattr(self, 'train_dataset', None), getattr(self, 'eval_dataset', None)]
        for dataset in datasets:
            if dataset is None:
                continue
            if isinstance(dataset, dict):
                yield from (ds for ds in dataset.values() if ds is not None)
            else:
                yield dataset

    def _build_rule_reward_index(self):
        pad_token_id = self.processing_class.pad_token_id
        num_rows = 0
        for dataset in self._iter_rule_reward_datasets():
            for row in dataset:
                input_ids = row.get('input_ids')
                if input_ids is None:
                    continue
                key = self._normalize_token_ids(input_ids, pad_token_id)
                if key and key not in self._ppo_rule_reward_kwargs_by_prompt:
                    self._ppo_rule_reward_kwargs_by_prompt[key] = self._row_extra_kwargs(row)
                    num_rows += 1
        if num_rows == 0:
            logger.warning('PPO rule-based rewards are enabled, but no prompt metadata was indexed from datasets.')

    def _prepare_rule_rewards(self, reward_funcs):
        args = self.args
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        reward_funcs = list(reward_funcs)

        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and reward_func in orms:
                reward_func_class = orms[reward_func]
                reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                reward_func_kwargs = {}
                for key in reward_func_args:
                    if key in ['self', 'args', 'kwargs']:
                        continue
                    if key in self._ppo_rule_reward_func_kwargs:
                        reward_func_kwargs[key] = self._ppo_rule_reward_func_kwargs[key]
                    elif hasattr(args, key):
                        reward_func_kwargs[key] = getattr(args, key)
                if 'tokenizer' in reward_func_args:
                    reward_func_kwargs['tokenizer'] = self.processing_class
                reward_funcs[i] = reward_func_class(**reward_func_kwargs)
            elif not callable(reward_func):
                raise ValueError(f'reward_function {reward_func} is not implemented in swift.plugin')

        self._ppo_rule_reward_funcs = reward_funcs
        self._ppo_rule_reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self._ppo_rule_reward_func_names.append(reward_func_name)

        if self._ppo_rule_reward_weight_values is not None:
            if len(self._ppo_rule_reward_weight_values) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(self._ppo_rule_reward_weight_values)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            reward_weights = self._ppo_rule_reward_weight_values
        else:
            reward_weights = [1.0] * len(reward_funcs)
        self._ppo_rule_reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        logger.info(f'Using PPO rule-based rewards: {self._ppo_rule_reward_func_names}, weights={reward_weights}')

    def _prepare_rule_reward_weighting(self):
        args = self.args
        self.reward_func_names = self._ppo_rule_reward_func_names
        self.reward_weights = self._ppo_rule_reward_weights.to(self.accelerator.device)
        self.reward_weighting_strategy = getattr(args, 'reward_weighting_strategy', 'static')
        self.reward_ema_decay = getattr(args, 'reward_ema_decay', 0.9)
        self.reward_ema_epsilon = getattr(args, 'reward_ema_epsilon', 0.01)
        self.reward_ema = None
        self.reward_variance_ema = None
        self.reward_mean_weight = getattr(args, 'reward_mean_weight', 1.0)
        self.reward_variance_weight = getattr(args, 'reward_variance_weight', 0.1)
        self.reward_lcb_k_factor = getattr(args, 'reward_lcb_k_factor', 1.0)
        self.reward_nonlinear_tau = getattr(args, 'reward_nonlinear_tau', 1.0)
        self.reward_sigmoid_k = getattr(args, 'reward_sigmoid_k', 10.0)
        self.reward_sigmoid_threshold = getattr(args, 'reward_sigmoid_threshold', 0.8)
        self.reward_gradient_k = getattr(args, 'reward_gradient_k', 5.0)
        self.reward_snr_alpha = getattr(args, 'reward_snr_alpha', 1.0)
        self.reward_stagnation_beta = getattr(args, 'reward_stagnation_beta', 1.0)
        self.reward_softplus_alpha = getattr(args, 'reward_softplus_alpha', 1.0)
        self.reward_softplus_k1 = getattr(args, 'reward_softplus_k1', 5.0)
        self.reward_softplus_k2 = getattr(args, 'reward_softplus_k2', 10.0)
        self.reward_softplus_use_shortfall = getattr(args, 'reward_softplus_use_shortfall', True)
        self.reward_softplus_use_cv_bonus = getattr(args, 'reward_softplus_use_cv_bonus', True)
        self.reward_softplus_use_stagnation = getattr(args, 'reward_softplus_use_stagnation', True)
        self.reward_softplus_use_sum = getattr(args, 'reward_softplus_use_sum', False)
        self.reward_prev_ema = None
        self.reward_weight_update_interval = max(int(getattr(args, 'reward_weight_update_interval', 1)), 1)
        self._reward_weight_calls = 0

        self.drbo_tau = getattr(args, 'drbo_tau', 0.0)
        self.drbo_mab_d = max(int(getattr(args, 'drbo_mab_d', 1)), 1)
        self.drbo_cumsum = None
        self.drbo_cum_count = None
        self.drbo_indic_count = None
        self.drbo_pre_actor_rewards = None
        self.drbo_total_sample = 0
        self.drbo_step = 0
        self._logged_softplus_params = False
        self._ppo_rule_reward_update_buffer = []
        self._ppo_rule_reward_update_buffer_size = 0
        self._ppo_rule_reward_update_batch_size = getattr(args, 'local_batch_size', None)
        if self._ppo_rule_reward_update_batch_size is None:
            per_device_batch = getattr(args, 'per_device_train_batch_size', 1)
            grad_accum = getattr(args, 'gradient_accumulation_steps', 1)
            self._ppo_rule_reward_update_batch_size = per_device_batch * grad_accum
        self._ppo_rule_reward_update_batch_size = max(int(self._ppo_rule_reward_update_batch_size), 1)

        if self.reward_weighting_strategy != 'static':
            logger.info(
                f'Using PPO adaptive reward weighting: strategy={self.reward_weighting_strategy}, '
                f'initial_weights={self.reward_weights.detach().cpu().tolist()}')

    def _record_rule_reward_metric(self, key: str, value: Any) -> None:
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        else:
            value = float(value)
        self._ppo_rule_reward_pending_metrics.setdefault(key, []).append(value)

    def _record_rule_reward_vector_metrics(self, prefix: str, values: torch.Tensor) -> None:
        values = values.detach().float().cpu()
        for i, name in enumerate(self.reward_func_names):
            self._record_rule_reward_metric(f'{prefix}/{name}', values[i])

    def _normalize_adaptive_reward_weights(self, adaptive_weights: torch.Tensor) -> torch.Tensor:
        adaptive_weights = torch.clamp(adaptive_weights.float(), min=self.reward_ema_epsilon)
        adaptive_weights = torch.nan_to_num(
            adaptive_weights, nan=self.reward_ema_epsilon, posinf=1e6, neginf=0.0)
        weight_sum = adaptive_weights.sum()
        if not torch.isfinite(weight_sum).item() or weight_sum.item() <= 0:
            adaptive_weights = torch.ones_like(adaptive_weights)
            weight_sum = adaptive_weights.sum()
        return adaptive_weights * adaptive_weights.numel() / weight_sum

    def _set_rule_reward_weights(self, reward_weights: torch.Tensor) -> None:
        self.reward_weights = reward_weights.detach()
        self._ppo_rule_reward_weights = self.reward_weights

    def _log_current_rule_reward_weights(self) -> None:
        self._record_rule_reward_vector_metrics('reward_weights', self.reward_weights)

    def _gather_rule_reward_updates(self, rewards_per_func: torch.Tensor) -> torch.Tensor:
        if getattr(self.accelerator, 'num_processes', 1) <= 1:
            return rewards_per_func
        return gather(rewards_per_func)

    def _queue_adaptive_reward_weight_update(self, rewards_per_func: torch.Tensor) -> None:
        if self.reward_weighting_strategy == 'static':
            return
        self._ppo_rule_reward_update_buffer.append(rewards_per_func.detach())
        self._ppo_rule_reward_update_buffer_size += rewards_per_func.shape[0]
        if self._ppo_rule_reward_update_buffer_size < self._ppo_rule_reward_update_batch_size:
            return

        buffered_rewards = torch.cat(self._ppo_rule_reward_update_buffer, dim=0)
        self._ppo_rule_reward_update_buffer.clear()
        self._ppo_rule_reward_update_buffer_size = 0
        buffered_rewards = self._gather_rule_reward_updates(buffered_rewards)
        self._update_adaptive_reward_weights(buffered_rewards)

    def _update_adaptive_reward_weights(self, rewards_per_func: torch.Tensor) -> None:
        if self.reward_weighting_strategy == 'static':
            return
        if not getattr(getattr(self, 'model', None), 'training', False):
            return

        device = rewards_per_func.device
        is_drbo = self.reward_weighting_strategy in ('drbo_inverse', 'drbo_delta')
        if not is_drbo:
            self._reward_weight_calls += 1
            should_update_weights = self._reward_weight_calls % self.reward_weight_update_interval == 0
            batch_mean_rewards = torch.nanmean(rewards_per_func, dim=0)
            if self.reward_ema is None:
                self.reward_ema = batch_mean_rewards.clone()
            else:
                self.reward_ema = (
                    self.reward_ema_decay * self.reward_ema.to(device)
                    + (1 - self.reward_ema_decay) * batch_mean_rewards
                )

            needs_variance = self.reward_weighting_strategy in (
                'adaptive_mean_variance', 'adaptive_lcb', 'adaptive_exploration',
                'adaptive_mean_std_sigmoid', 'adaptive_mean_variance_sigmoid',
                'adaptive_mean_variance_sigmoid_v2', 'adaptive_mean_variance_sigmoid_v3',
                'adaptive_mean_variance_softplus',
            )
            if needs_variance:
                batch_mean_squared = torch.nanmean(rewards_per_func**2, dim=0)
                batch_variance = torch.clamp(batch_mean_squared - batch_mean_rewards**2, min=0.0)
                if self.reward_variance_ema is None:
                    self.reward_variance_ema = batch_variance.clone()
                else:
                    self.reward_variance_ema = (
                        self.reward_ema_decay * self.reward_variance_ema.to(device)
                        + (1 - self.reward_ema_decay) * batch_variance
                    )

            needs_delta = self.reward_weighting_strategy in (
                'adaptive_mean_variance_sigmoid_v2',
                'adaptive_mean_variance_sigmoid_v3',
                'adaptive_mean_variance_softplus',
            )
            if needs_delta:
                if self.reward_prev_ema is None:
                    delta_mu = torch.zeros_like(self.reward_ema)
                else:
                    delta_mu = torch.clamp(self.reward_ema - self.reward_prev_ema.to(device), min=-1.0, max=1.0)
                self.reward_prev_ema = self.reward_ema.clone().detach()

            if not should_update_weights:
                return

        if self.reward_weighting_strategy == 'adaptive_shortfall':
            adaptive_weights = 1.0 / torch.clamp(self.reward_ema, min=self.reward_ema_epsilon)
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)

        elif self.reward_weighting_strategy == 'adaptive_mean_variance':
            mean_component = 1.0 - self.reward_ema
            variance_component = self.reward_variance_ema
            adaptive_weights = (
                self.reward_mean_weight * mean_component + self.reward_variance_weight * variance_component)
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_mean_component', mean_component)
            self._record_rule_reward_vector_metrics('reward_variance_component', variance_component)

        elif self.reward_weighting_strategy == 'adaptive_lcb':
            reward_std_ema = torch.sqrt(torch.clamp(self.reward_variance_ema, min=0.0))
            lcb = self.reward_ema - self.reward_lcb_k_factor * reward_std_ema
            adaptive_weights = 1.0 / torch.clamp(lcb, min=self.reward_ema_epsilon)
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_std_ema', reward_std_ema)
            self._record_rule_reward_vector_metrics('reward_lcb', lcb)

        elif self.reward_weighting_strategy == 'adaptive_exploration':
            reward_std_ema = torch.sqrt(torch.clamp(self.reward_variance_ema, min=0.0))
            mean_complement = torch.clamp(1.0 - self.reward_ema, min=self.reward_ema_epsilon)
            tau = max(self.reward_nonlinear_tau, self.reward_ema_epsilon)
            adaptive_weights = mean_complement * torch.exp(reward_std_ema / tau)
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_std_ema', reward_std_ema)
            self._record_rule_reward_vector_metrics('reward_mean_complement', mean_complement)

        elif self.reward_weighting_strategy == 'adaptive_mean_std_sigmoid':
            reward_std_ema = torch.sqrt(torch.clamp(self.reward_variance_ema, min=0.0))
            mean_component = 1.0 - self.reward_ema
            sigmoid_gate = torch.sigmoid(self.reward_sigmoid_k * (self.reward_sigmoid_threshold - self.reward_ema))
            adaptive_weights = mean_component + sigmoid_gate * reward_std_ema
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_std_ema', reward_std_ema)
            self._record_rule_reward_vector_metrics('reward_mean_component', mean_component)
            self._record_rule_reward_vector_metrics('reward_sigmoid_gate', sigmoid_gate)

        elif self.reward_weighting_strategy == 'adaptive_mean_variance_sigmoid':
            mean_component = 1.0 - self.reward_ema
            sigmoid_gate = torch.sigmoid(self.reward_sigmoid_k * (self.reward_sigmoid_threshold - self.reward_ema))
            variance_component = self.reward_variance_ema
            adaptive_weights = mean_component + sigmoid_gate * variance_component
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_variance_component', variance_component)
            self._record_rule_reward_vector_metrics('reward_mean_component', mean_component)
            self._record_rule_reward_vector_metrics('reward_sigmoid_gate', sigmoid_gate)

        elif self.reward_weighting_strategy == 'adaptive_mean_variance_sigmoid_v2':
            reward_std_ema = torch.sqrt(torch.clamp(self.reward_variance_ema, min=0.0))
            normalized_snr = torch.exp(reward_std_ema / (self.reward_ema + self.reward_ema_epsilon))
            dynamic_tau = self.reward_ema.mean()
            sigmoid_gate = torch.sigmoid(self.reward_sigmoid_k * (dynamic_tau - self.reward_ema))
            gradient_gate = torch.sigmoid(self.reward_gradient_k * (1.0 - delta_mu))
            adaptive_weights = sigmoid_gate * normalized_snr * gradient_gate
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_std_ema', reward_std_ema)
            self._record_rule_reward_metric('reward_dynamic_tau', dynamic_tau)
            self._record_rule_reward_vector_metrics('reward_sigmoid_gate', sigmoid_gate)
            self._record_rule_reward_vector_metrics('reward_normalized_snr', normalized_snr)
            self._record_rule_reward_vector_metrics('reward_delta_mu', delta_mu)
            self._record_rule_reward_vector_metrics('reward_gradient_gate', gradient_gate)

        elif self.reward_weighting_strategy == 'adaptive_mean_variance_sigmoid_v3':
            reward_std_ema = torch.sqrt(torch.clamp(self.reward_variance_ema, min=0.0))
            normalized_snr = torch.exp(
                self.reward_snr_alpha * reward_std_ema / (self.reward_ema + self.reward_ema_epsilon))
            dynamic_tau = self.reward_ema.mean()
            sigmoid_gate = torch.sigmoid(self.reward_sigmoid_k * (dynamic_tau - self.reward_ema))
            stagnation_penalty = torch.exp(-self.reward_stagnation_beta * delta_mu)
            adaptive_weights = sigmoid_gate * normalized_snr * stagnation_penalty
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_std_ema', reward_std_ema)
            self._record_rule_reward_metric('reward_dynamic_tau', dynamic_tau)
            self._record_rule_reward_vector_metrics('reward_sigmoid_gate', sigmoid_gate)
            self._record_rule_reward_vector_metrics('reward_normalized_snr', normalized_snr)
            self._record_rule_reward_vector_metrics('reward_delta_mu', delta_mu)
            self._record_rule_reward_vector_metrics('reward_stagnation_penalty', stagnation_penalty)

        elif self.reward_weighting_strategy == 'adaptive_mean_variance_softplus':
            eps = 1e-3
            delta = 1e-3
            if not self._logged_softplus_params:
                logger.info(
                    '[PPO adaptive_mean_variance_softplus] params: eps=%.0e delta=%.0e k1=%.1f k2=%.1f '
                    'alpha=%.1f reward_weight_update_interval=%d use_shortfall=%s combine=%s',
                    eps, delta, self.reward_softplus_k1, self.reward_softplus_k2, self.reward_softplus_alpha,
                    self.reward_weight_update_interval, self.reward_softplus_use_shortfall,
                    'sum' if self.reward_softplus_use_sum else 'product')
                self._logged_softplus_params = True
            reward_std_ema = torch.sqrt(torch.clamp(self.reward_variance_ema, min=0.0))
            mu_bar = self.reward_ema.mean()
            if self.reward_softplus_use_shortfall:
                shortfall_gate = delta + torch.nn.functional.softplus(
                    self.reward_softplus_k1 * (mu_bar - self.reward_ema))
            else:
                shortfall_gate = torch.ones_like(self.reward_ema)
            if self.reward_softplus_use_cv_bonus:
                cv_bonus = 1.0 + self.reward_softplus_alpha * reward_std_ema / (self.reward_ema + eps)
            else:
                cv_bonus = torch.ones_like(self.reward_ema)
            if self.reward_softplus_use_stagnation:
                stagnation_gate = delta + torch.nn.functional.softplus(-self.reward_softplus_k2 * delta_mu)
            else:
                stagnation_gate = torch.ones_like(self.reward_ema)
            if self.reward_softplus_use_sum:
                adaptive_weights = shortfall_gate + cv_bonus + stagnation_gate
            else:
                adaptive_weights = shortfall_gate * cv_bonus * stagnation_gate
            self._set_rule_reward_weights(self._normalize_adaptive_reward_weights(adaptive_weights).to(device))
            self._record_rule_reward_vector_metrics('reward_ema', self.reward_ema)
            self._record_rule_reward_vector_metrics('reward_variance_ema', self.reward_variance_ema)
            self._record_rule_reward_vector_metrics('reward_std_ema', reward_std_ema)
            self._record_rule_reward_metric('reward_dynamic_tau', mu_bar)
            self._record_rule_reward_vector_metrics('reward_shortfall_gate', shortfall_gate)
            self._record_rule_reward_vector_metrics('reward_cv_bonus', cv_bonus)
            self._record_rule_reward_vector_metrics('reward_delta_mu', delta_mu)
            self._record_rule_reward_vector_metrics('reward_stagnation_gate', stagnation_gate)

        elif self.reward_weighting_strategy in ('drbo_inverse', 'drbo_delta'):
            num_funcs = rewards_per_func.shape[1]
            batch_count = (~torch.isnan(rewards_per_func)).sum(dim=0).float().cpu()
            batch_sum = torch.nansum(rewards_per_func, dim=0).cpu()
            if self.drbo_cumsum is None:
                self.drbo_cumsum = torch.zeros(num_funcs, dtype=torch.float32)
                self.drbo_cum_count = torch.zeros(num_funcs, dtype=torch.float32)
                self.drbo_indic_count = torch.zeros(num_funcs, dtype=torch.float32)
                self.drbo_pre_actor_rewards = torch.zeros(num_funcs, dtype=torch.float32)

            pre_actor_rewards = self.drbo_cumsum.clone()
            self.drbo_cumsum += batch_sum
            self.drbo_cum_count += batch_count
            self.drbo_total_sample += rewards_per_func.shape[0]
            self.drbo_step += 1

            valid_func_mask = batch_count > 0
            cum_mean = self.drbo_cumsum / self.drbo_cum_count.clamp(min=1.0)
            num_funcs_float = float(num_funcs)
            if self.drbo_step % self.drbo_mab_d == 0:
                self.drbo_indic_count += valid_func_mask.float()
                if self.reward_weighting_strategy == 'drbo_inverse':
                    prob_tmp_avg = (cum_mean + 1.0) / num_funcs_float
                    log_total = torch.log(torch.tensor(max(self.drbo_total_sample, 1), dtype=torch.float32))
                    ucb_bonus = torch.sqrt(2.0 * log_total / self.drbo_indic_count.clamp(min=1.0))
                    prob_tmp = prob_tmp_avg + ucb_bonus
                    min_avg = prob_tmp_avg.min()
                    denom = prob_tmp - self.drbo_tau * min_avg
                    z = torch.where(valid_func_mask & (prob_tmp != 0), 1.0 / denom, torch.zeros_like(prob_tmp))
                    self._record_rule_reward_vector_metrics('reward_drbo_prob_avg', prob_tmp_avg)
                    self._record_rule_reward_vector_metrics('reward_drbo_ucb', ucb_bonus)
                else:
                    drbo_delta = (cum_mean - pre_actor_rewards) / num_funcs_float
                    step = max(self.drbo_step, 1)
                    min_delta = drbo_delta.min()
                    exponent = (drbo_delta - self.drbo_tau * min_delta) / step
                    z = torch.where(
                        valid_func_mask & (drbo_delta != 0), torch.exp(exponent), torch.zeros_like(drbo_delta))
                    self._record_rule_reward_vector_metrics('reward_drbo_delta', drbo_delta)

                self.drbo_pre_actor_rewards = self.drbo_cumsum.clone().detach()
                z_sum = z.sum()
                if z_sum > 0:
                    normalized = z / z_sum * num_funcs
                    keep_mask = z == 0
                    if keep_mask.any():
                        normalized[keep_mask] = self.reward_weights.detach().cpu()[keep_mask]
                    self._set_rule_reward_weights(normalized.to(device))

            self._record_rule_reward_vector_metrics('reward_drbo_cum_mean', cum_mean)
            self._record_rule_reward_vector_metrics('reward_drbo_indic_count', self.drbo_indic_count)

        else:
            raise ValueError(
                f"Unknown reward_weighting_strategy: {self.reward_weighting_strategy}. "
                f"Supported strategies: 'static', 'adaptive_shortfall', 'adaptive_mean_variance', "
                f"'adaptive_lcb', 'adaptive_exploration', 'adaptive_mean_std_sigmoid', "
                f"'adaptive_mean_variance_sigmoid', 'adaptive_mean_variance_sigmoid_v2', "
                f"'adaptive_mean_variance_sigmoid_v3', 'adaptive_mean_variance_softplus', "
                f"'drbo_inverse', 'drbo_delta'")

        self._log_current_rule_reward_weights()

    def _get_rule_reward_kwargs(self, query_responses: torch.Tensor, context_length: int) -> Dict[str, List[Any]]:
        pad_token_id = self.processing_class.pad_token_id
        rows = []
        for query_ids in query_responses[:, :context_length]:
            key = self._normalize_token_ids(query_ids.detach().cpu(), pad_token_id)
            rows.append(self._ppo_rule_reward_kwargs_by_prompt.get(key, {}))
        return RowPreprocessor.rows_to_batched(rows)

    def _get_response_token_ids(self, response_ids: torch.Tensor) -> tuple:
        pad_token_id = self.processing_class.pad_token_id
        eos_token_id = self.processing_class.eos_token_id
        response_token_ids = []
        is_truncated = []
        for ids in response_ids.detach().cpu().tolist():
            clean_ids = []
            reached_stop = False
            for token in ids:
                if pad_token_id is not None and token == pad_token_id:
                    reached_stop = True
                    break
                clean_ids.append(token)
                if eos_token_id is not None and token == eos_token_id:
                    reached_stop = True
                    break
            response_token_ids.append(clean_ids)
            is_truncated.append(not reached_stop and len(clean_ids) >= response_ids.shape[1])
        return response_token_ids, is_truncated

    def _compute_rule_reward_output(
            self, query_responses: torch.Tensor, pad_token_id: int, context_length: int,
            update_reward_weights: bool = True):
        device = query_responses.device
        response_ids = query_responses[:, context_length:]
        completions = self.processing_class.batch_decode(response_ids, skip_special_tokens=True)
        reward_kwargs = self._get_rule_reward_kwargs(query_responses, context_length)
        response_token_ids, is_truncated = self._get_response_token_ids(response_ids)
        reward_kwargs.update({
            'response_token_ids': response_token_ids,
            'is_truncated': is_truncated,
            'trainer_state': self.state,
        })

        rewards_per_func = torch.zeros(
            (query_responses.shape[0], len(self._ppo_rule_reward_funcs)), dtype=torch.float32, device=device)
        for i, (reward_func, reward_func_name) in enumerate(
                zip(self._ppo_rule_reward_funcs, self._ppo_rule_reward_func_names)):
            reward_args = self._get_reward_call_parameters(reward_func)
            if ('solution' in reward_args
                    and ('solution' not in reward_kwargs or any(s is None for s in reward_kwargs['solution']))):
                raise ValueError(f"PPO rule reward '{reward_func_name}' requires the `solution` column. "
                                 'Please keep `remove_unused_columns=false` and use a math dataset with solutions.')
            output_reward_func = reward_func(completions, **reward_kwargs)
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            reward_tensor = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            if reward_tensor.shape[0] != query_responses.shape[0]:
                raise ValueError(
                    f"Reward function '{reward_func_name}' returned {reward_tensor.shape[0]} rewards "
                    f"but expected {query_responses.shape[0]}."
                )
            rewards_per_func[:, i] = reward_tensor

        reward_weights = self._ppo_rule_reward_weights.to(device)
        scores = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)
        if pad_token_id is None:
            padding_mask = torch.zeros_like(response_ids, dtype=torch.bool)
        else:
            padding_mask = response_ids == pad_token_id
        sequence_lengths = self._first_true_indices(padding_mask) - 1 + context_length
        reward_logits = torch.zeros(
            (query_responses.shape[0], query_responses.shape[1], 1), dtype=scores.dtype, device=device)
        reward_logits[torch.arange(reward_logits.shape[0], device=device), sequence_lengths, 0] = scores
        # PPO computes rewards in rollout chunks; update after scoring so chunk size does not change current rewards.
        if update_reward_weights:
            # Log per-dimension rewards for all strategies (including static, which otherwise
            # only records the weighted sum `objective/scores`). Mean over the batch per func.
            self._record_rule_reward_vector_metrics('reward', rewards_per_func.nanmean(dim=0))
            self._queue_adaptive_reward_weight_update(rewards_per_func)
        return reward_logits, scores, sequence_lengths

    @contextmanager
    def _patch_rule_based_reward(self):
        if not self._ppo_rule_reward_enabled:
            yield
            return

        train_globals = HFPPOTrainer.train.__globals__
        generate_globals = HFPPOTrainer.generate_completions.__globals__
        original_train_get_reward = train_globals['get_reward']
        original_generate_get_reward = generate_globals['get_reward']

        def patched_get_reward(model, query_responses, pad_token_id, context_length):
            if model is self.reward_model:
                return self._compute_rule_reward_output(
                    query_responses,
                    pad_token_id,
                    context_length,
                    update_reward_weights=not self._ppo_rule_reward_in_generate)
            original_get_reward = (
                original_generate_get_reward if self._ppo_rule_reward_in_generate else original_train_get_reward)
            return original_get_reward(model, query_responses, pad_token_id, context_length)

        train_globals['get_reward'] = patched_get_reward
        generate_globals['get_reward'] = patched_get_reward
        try:
            yield
        finally:
            train_globals['get_reward'] = original_train_get_reward
            generate_globals['get_reward'] = original_generate_get_reward

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._ppo_rule_reward_pending_metrics:
            logs = dict(logs)
            for key, values in self._ppo_rule_reward_pending_metrics.items():
                logs[key] = sum(values) / len(values)
            self._ppo_rule_reward_pending_metrics.clear()
        return super().log(logs, *args, **kwargs)

    def train(self, *args, **kwargs):
        # remove args that are not needed for the HFPPOTrainer
        resume_from_checkpoint = args[0] if args else kwargs.get('resume_from_checkpoint')
        if resume_from_checkpoint and not self.args.resume_only_model:
            logger.warning('TRL PPOTrainer does not restore optimizer/value/trainer state from resume_from_checkpoint. '
                           'Use `--resume_only_model true` to initialize the policy from a checkpoint.')
        with self._patch_rule_based_reward(), self._patch_vllm_generation():
            super().train()

    def _save_checkpoint(self, *args, **kwargs):
        kwargs.pop('metrics', None)

        backup_model = self.model
        try:
            # Unwrap model if needed
            self.model = self.accelerator.unwrap_model(self.model)
            return super()._save_checkpoint(*args, **kwargs)
        finally:
            self.model = backup_model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # https://github.com/huggingface/trl/issues/2122
        backup_model = self.model

        # Unwrap model if needed to access the policy
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.model = unwrapped_model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {
                name.removeprefix('policy.'): param
                for name, param in state_dict.items() if name.startswith('policy.')
            }

        super()._save(output_dir, state_dict)

    def _prepare_gradient_checkpointing(self, model):
        # Be consistent with TRL
        # models = list(set([self.model.policy, self.model.value_model]))
        # for model in models:
        #     SwiftMixin._prepare_gradient_checkpointing(self, model)
        pass

    def generate_completions(self, *args, **kwargs):
        if self._swift_no_eval_dataset or self.eval_dataset is None:
            return
        old_in_generate = self._ppo_rule_reward_in_generate
        self._ppo_rule_reward_in_generate = True
        try:
            with self._patch_rule_based_reward():
                return super().generate_completions(*args, **kwargs)
        finally:
            self._ppo_rule_reward_in_generate = old_in_generate
