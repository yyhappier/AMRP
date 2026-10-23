# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import nullcontext
from typing import List, Optional, Union

from swift.llm import safe_snapshot_download
from swift.plugin import Tuner, extra_tuners
from swift.tuners import Swift
from swift.utils import get_logger, get_model_parameter_info
from swift.utils.utils import disable_deepspeed_zero3
from ..argument import BaseArguments, RLHFArguments
from ..argument.rlhf_args import rlhf_grpo_like_types
from ..model import HfConfigFactory
from .kto import prepare_kto_dataset
from .sft import SwiftSft

logger = get_logger()


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    @staticmethod
    def _get_model_task_type(model_dir):
        task_type = None
        num_labels = None
        if os.path.exists(os.path.join(model_dir, 'args.json')):
            model_args = BaseArguments.from_pretrained(model_dir)
            if hasattr(model_args, 'task_type'):
                task_type = model_args.task_type
            if hasattr(model_args, 'num_labels'):
                num_labels = model_args.num_labels
            if task_type == 'seq_cls' and num_labels is None:
                num_labels = 1
        else:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            if hasattr(model_config, 'architectures') and model_config.architectures:
                if any('sequenceclassification' in arch.lower() for arch in model_config.architectures):
                    task_type = 'seq_cls'
                    num_labels = getattr(model_config, 'num_labels', None) or 1

            if task_type is None:
                if hasattr(model_config, 'num_labels'):
                    num_labels = model_config.num_labels
                    # PretrainedConfig default num_labels = 2
                    if num_labels == 1:
                        task_type = 'seq_cls'
        return task_type, num_labels

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):
        from swift.llm.infer.utils import prepare_adapter
        args = self.args
        origin_key = origin_key or key
        model_id_or_path = getattr(args, f'{key}_model')
        if model_id_or_path is None:
            return
        if isinstance(model_id_or_path, list):
            # value model in PPO
            model_id_or_path = model_id_or_path[0]

        model_dir = safe_snapshot_download(
            model_id_or_path=model_id_or_path,
            revision=model_revision,
            download_model=False,
            use_hf=args.use_hf,
            hub_token=args.hub_token,
        )
        task_type, num_labels = self._get_model_task_type(model_dir)
        if origin_key == 'value' and args.rlhf_type == 'ppo' and args.reward_funcs:
            if task_type is not None and task_type != 'seq_cls':
                raise ValueError(f'PPO rule-based value model expects a seq_cls/causal backbone, got {task_type}.')
            if task_type == 'seq_cls' and num_labels not in {None, 1}:
                raise ValueError('PPO rule-based value model requires a scalar value head; '
                                 f'got num_labels={num_labels}.')
            # Use the supplied backbone as PPO value model and add a scalar score head.
            task_type, num_labels = 'seq_cls', 1
        context = nullcontext()
        if key == 'teacher' and args.teacher_deepspeed:
            if args.teacher_deepspeed.get('zero_optimization', {}).get('stage') != 3:
                context = disable_deepspeed_zero3()
        with context:
            model, processor = args.get_model_processor(
                model=model_id_or_path,
                model_type=model_type,
                model_revision=model_revision,
                task_type=task_type,
                num_labels=num_labels)

        adapters = args.adapters if key == 'ref' else args.reward_adapters
        model = prepare_adapter(args, model, adapters)
        if origin_key in {'ref', 'reward', 'teacher'}:
            if self.args.sequence_parallel_size > 1:
                from swift.trainers.sequence_parallel import sequence_parallel
                sequence_parallel.prepare(
                    self.args.sequence_parallel_size, model, processor, padding_free=args.padding_free)
            model.requires_grad_(False).eval()
        else:
            model = self.prepare_model(args, model, task_type=task_type)
            logger.info(f'value_model: {model}')
            model_parameter_info = get_model_parameter_info(model)
            self.train_msg['value_model_parameter_info'] = model_parameter_info
            logger.info(f'value_model_parameter_info: {model_parameter_info}')

        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        return model, processor

    def _prepare_model_tokenizer(self):
        # prepare ref/reward/value model
        args = self.args
        # Handle ref and value models
        for key in ['ref', 'value', 'teacher']:
            setattr(self, f'{key}_model', None)
            if key == 'ref' and args.rlhf_type == 'gkd':
                continue
            if key == 'value' and args.rlhf_type != 'ppo':
                continue
            if key == 'teacher' and args.rlhf_type != 'gkd':
                continue
            model_key = 'reward' if key == 'value' else key
            model_type = getattr(args, f'{model_key}_model_type')
            model_revision = getattr(args, f'{model_key}_model_revision')
            if key == 'value':
                model_type = model_type[0] if model_type else None
                model_revision = model_revision[0] if model_revision else None

            result = self._prepare_single_model(model_key, key, model_type, model_revision)
            if result is not None:
                model, _ = result
                setattr(self, f'{key}_model', model)

        # Handle reward model(s)
        self.reward_model = None
        if (hasattr(args, 'reward_model') and args.reward_model is not None
                and not (args.rlhf_type == 'ppo' and args.reward_funcs)):
            rms = args.reward_model if isinstance(args.reward_model, list) else [args.reward_model]
            num_rms = len(rms)
            rm_types = args.reward_model_type if args.reward_model_type else [None] * num_rms
            rm_revisions = args.reward_model_revision if args.reward_model_revision else [None] * num_rms
            assert len(rms) == len(rm_types) == len(rm_revisions)

            self.reward_model = []
            if args.rlhf_type in rlhf_grpo_like_types:
                self.reward_template = []

            # Cache loaded reward models by (path, type, revision) to avoid
            # loading the same checkpoint multiple times when several plugins
            # share the same model (e.g. armo_helpfulness/correctness/coherence).
            _rm_cache: dict = {}

            for reward_model_path, rm_type, rm_revision in zip(rms, rm_types, rm_revisions):
                cache_key = (reward_model_path, rm_type, rm_revision)
                if cache_key in _rm_cache:
                    model, processor = _rm_cache[cache_key]
                else:
                    args.reward_model = reward_model_path  # Temporarily set for prepare_single_model
                    result = self._prepare_single_model('reward', None, rm_type, rm_revision)
                    args.reward_model = rms  # Restore original value
                    if result is None:
                        continue
                    model, processor = result
                    _rm_cache[cache_key] = (model, processor)

                self.reward_model.append(model)

                if args.rlhf_type in rlhf_grpo_like_types:
                    if cache_key in _rm_cache and self.reward_template:
                        # Reuse a copy of the already-built template for this model.
                        import copy
                        reward_template = copy.copy(self.reward_template[-1])
                    else:
                        reward_template = self.args.get_template(processor, processor.model_meta.template)
                        if reward_template.use_model:
                            reward_template.model = model
                    self.reward_template.append(reward_template)

            if args.rlhf_type not in rlhf_grpo_like_types and self.reward_model:
                assert len(self.reward_model) <= 1
                self.reward_model = self.reward_model[0]

        super()._prepare_model_tokenizer()

    @classmethod
    def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):
        model = super().prepare_model(args, model, template=template, train_dataset=train_dataset, task_type=task_type)
        if args.ref_adapters:
            if args.train_type in extra_tuners:
                tuner: Tuner = extra_tuners[args.train_type]
            else:
                tuner = Swift
            assert len(args.ref_adapters) == 1, f'args.ref_adapters: {args.ref_adapters}'
            model = tuner.from_pretrained(model, args.ref_adapters[0], adapter_name='ref_adapter')
            assert args.rlhf_type in {'dpo', 'kto', 'grpo', 'gdpo'}, (
                'Currently, only DPO, KTO, GRPO, and GDPO support `ref_adapters`.')
            args.training_args.ref_adapter_name = 'ref_adapter'
        return model

    def _prepare_template(self) -> None:
        args = self.args
        super()._prepare_template()
        model_mapping = {'kto': 'kto', 'gkd': 'gkd', 'ppo': 'pt', 'grpo': 'train', 'gdpo': 'train'}
        self.template.set_mode(model_mapping.get(args.rlhf_type, 'rlhf'))

        if args.rlhf_type == 'ppo':
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _prepare_chord_sft_dataset(self):
        from ..dataset import load_dataset
        from swift.llm.dataset.loader import DatasetLoader

        # prepare expert sft dataset for chord
        args = self.args
        assert hasattr(args, 'chord_sft_dataset') and args.chord_sft_dataset
        dataset_kwargs = args.get_dataset_kwargs()
        chord_sft_datasets = []
        # TODO: validatition
        chord_sft_dataset, _ = load_dataset(
            args.chord_sft_dataset, split_dataset_ratio=0, shuffle=args.dataset_shuffle, **dataset_kwargs)
        chord_sft_dataset, _ = self._encode_dataset(chord_sft_dataset, None, pre_process=True)
        chord_sft_datasets.append(chord_sft_dataset)
        chord_sft_dataset = DatasetLoader._concat_datasets(chord_sft_datasets)
        datasets = [chord_sft_dataset, None]
        datasets = self._post_process_datasets(datasets)
        return datasets

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        for key in ['ref', 'reward', 'value', 'teacher']:
            key = f'{key}_model'
            model = getattr(self, key, None)
            if model or self.args.rlhf_type == 'ppo' and key != 'teacher_model':
                trainer_kwargs[key] = model
        if hasattr(self, 'reward_template'):
            trainer_kwargs['reward_template'] = self.reward_template
        if self.args.rlhf_type in ['grpo', 'gdpo', 'gkd']:
            trainer_kwargs['vllm_client'] = self.args.vllm_client
        if self.args.rlhf_type in rlhf_grpo_like_types or (self.args.rlhf_type == 'ppo' and self.args.reward_funcs):
            trainer_kwargs['reward_funcs'] = self.args.reward_funcs
            if self.args.rlhf_type == 'ppo':
                trainer_kwargs['reward_weights'] = self.args.reward_weights
                trainer_kwargs['reward_func_kwargs'] = self._get_ppo_rule_reward_func_kwargs()
        if self.args.rlhf_type in rlhf_grpo_like_types:
            if self.args.chord_sft_dataset:
                trainer_kwargs['chord_sft_dataset'], _ = self._prepare_chord_sft_dataset()
        if self.args.rlhf_type == 'gkd' and self.args.teacher_deepspeed:
            trainer_kwargs['teacher_deepspeed_config'] = self.args.teacher_deepspeed
        return trainer_kwargs

    def _get_ppo_rule_reward_func_kwargs(self):
        import inspect
        from swift.plugin import orms

        reward_func_kwargs = {}
        for reward_func in self.args.reward_funcs:
            if isinstance(reward_func, str) and reward_func in orms:
                reward_func_args = inspect.signature(orms[reward_func].__init__).parameters
                for key in reward_func_args:
                    if key not in ['self', 'args', 'kwargs'] and hasattr(self.args, key):
                        reward_func_kwargs[key] = getattr(self.args, key)
        return reward_func_kwargs


def rlhf_main(args: Optional[Union[List[str], RLHFArguments]] = None):
    return SwiftRLHF(args).main()
