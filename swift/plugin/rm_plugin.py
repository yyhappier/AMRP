import re
import textwrap
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List

import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.utils import get_logger

if TYPE_CHECKING:
    from swift.llm.infer.protocol import ChatCompletionResponse

logger = get_logger()


class DefaultRMPlugin:
    """
    Default Reward Model Plugin

    This class implements the default processing logic for reward models.
    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class GenRMPlugin(DefaultRMPlugin):

    def __init__(self, model, template):
        """
        Generative Reward Model Plugin Example.

        This method sets up the reward model plugin by initializing the PtEngine for efficient inference,
        configuring the request parameters, and defining the system prompt that guides the reward model in
        evaluating responses.

        Args:
            model (torch.nn.Module): The generative reward model.
            template (Template): The template used for encoding input data.
    """

        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig()  # customise your request config here
        self.system = textwrap.dedent("""
            Based on the dialogue history, analyze in detail whether the model's response is accurate, complete, and relevant.
            Assign a reward score between 0 and 1, where 0 indicates completely incorrect and 1 indicates fully correct.
            Before finishing your response, please assign a reward using the following format:

            Reward: {reward}

            For example:
            Reward: 0.85
        """)  # noqa

    def __call__(self, inputs, **kwargs):
        """
        Compute reward scores for the provided inputs.

        This method processes each input by converting dialogue messages into a query, sending the query to the
        reward model for inference, and extracting the reward scores from the model's responses. The final reward
        for each input is the average of all extracted scores.
        Args:
            inputs (List[Dict]): A list of input requests. Each input request is a dictionary containing:
                - 'messages' (List[Dict]): messages from the training model. Each message dictionary includes:
                    - 'role' (str): The role of the speaker (e.g., 'user', 'assistant').
                    - 'content' (str): The content of the message.
                - Additional dataset columns as key-value pairs (e.g., 'solutions', 'images').
        Returns:
            torch.Tensor: A tensor containing the average reward scores for each input. The tensor has a shape of (N,),
            where N is the number of input requests.
        """

        rm_inputs = self.prepare_rm_inputs(inputs)
        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        rewards = self.compute_rewards(results)
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict]) -> List[Dict]:
        """
        Prepare inputs for the reward model by converting messages into queries.

        Args:
            inputs (List[Dict]): A list of input requests.

        Returns:
            List[Dict]: Processed inputs for the reward model.
        """
        rm_inputs = []
        for idx, infer_request in enumerate(inputs):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)

            # Extract and convert messages to a single query string
            messages = rm_infer_request.get('messages')
            query = self.messages_to_query(messages)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': query}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        """
        Extract the reward score from the model's output.

        Args:
            model_output (str): The model's output string, expected to follow the format "Reward: {reward}".

        Returns:
            float: The extracted reward score.

        Raises:
            ValueError: If the reward score cannot be extracted or the format is incorrect.
        """
        match = re.search(r'Reward:\s*([0-1](?:\.\d+)?)', model_output)
        if match:
            return float(match.group(1))
        else:
            logger.warning("Unable to extract reward score from the model's output, set reward to 0")
            return None

    @staticmethod
    def messages_to_query(messages):
        """
        Compress a list of message dictionaries into a single query string.

        Args:
            messages (list[dict]): A list of message dictionaries, each containing:
                - 'role' (str): The role of the speaker (e.g., 'user', 'assistant').
                - 'content' (str): The content of the message.

        Returns:
            str: A single string that concatenates all messages in a formatted manner.

        Example:
            >>> messages = [
            ...     {'role': 'user', 'content': 'Hello, how are you?'},
            ...     {'role': 'assistant', 'content': 'I am fine, thank you! How can I assist you today?'},
            ...     {'role': 'user', 'content': 'Can you help me with my homework?'}
            ... ]
            >>> print(messages_to_query(messages))
            User: Hello, how are you?
            Assistant: I am fine, thank you! How can I assist you today?
            User: Can you help me with my homework?
        """
        # Initialize an empty list to hold formatted messages
        formatted_messages = []

        # Define a mapping for role capitalization if needed
        role_mapping = {
            'user': 'User',
            'assistant': 'Assistant',
            'system': 'System'
            # Add more roles here as needed
        }

        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f'Each message must be a dictionary. Found {type(message)} at index {idx}.')

            # Extract 'role' and 'content' from each message
            role = message.get('role')
            content = message.get('content')
            if not content:
                continue

            # Capitalize the role using the mapping, default to capitalized original role
            role_formatted = role_mapping.get(role.lower(), role.capitalize())

            # Append the formatted message to the list
            formatted_messages.append(f'{role_formatted}: {content}')

        # Join all formatted messages with newline characters
        query = '\n'.join(formatted_messages)

        return query

    def compute_rewards(self, results: List['ChatCompletionResponse']) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List['ChatCompletionResponse']): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


class SteerLMRMPlugin(DefaultRMPlugin):
    """
    Plugin for Llama3-70B-SteerLM-RM (NeMo → HF converted).

    SteerLM RM is trained with a label placeholder appended after the assistant
    response. The regression head scores the hidden state at the final token.
    Without the placeholder the model receives an out-of-distribution input.

    Input format (standard Llama-3 chat template, no label placeholder needed):
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>

    Per HelpSteer2 paper Sec.3: "a linear layer that converts the final layer
    representation of the end-of-response token into five scalar values."
    The RM scores the last token of the assistant response directly.

    Output dimensions (indices 0-3 are unused padding):
        4: helpfulness  5: correctness  6: coherence
        7: complexity   8: verbosity

    dim_indices: which dimensions to sum into the final scalar reward.
    Default [4, 5, 6] = helpfulness + correctness + coherence.
    """

    # Sentinel label appended to every input so the model scores at the right position
    def __init__(self, model, template, dim_indices: List[int] = None):
        super().__init__(model, template)
        self.dim_indices = dim_indices if dim_indices is not None else [4, 5, 6]
        # HF LlamaForSequenceClassification.score is Linear(hidden, num_labels, bias=False).
        # NeMo rm_head has a bias term saved as "score.bias" in the checkpoint.
        # HF from_pretrained ignores unexpected keys, so score.bias is silently dropped.
        # We fix this by replacing the score layer with a bias=True Linear and injecting
        # both weight and bias from the checkpoint's _metadata or the model's named_buffers.
        import torch.nn as nn
        score = getattr(model, 'score', None)
        if score is not None and not getattr(score, 'bias', None) is not None:
            # Look for bias in all named buffers/parameters (from_pretrained may have
            # stored it as a buffer if the user added it, or it may be in unexpected_keys)
            bias_val = None
            for name, buf in model.named_buffers():
                if name == 'score.bias':
                    bias_val = buf
                    break
            # Also try hf_device_map or internal state
            if bias_val is None:
                # Try to reload just the bias from the checkpoint index
                try:
                    import json, os
                    model_dir = model.config._name_or_path
                    for idx_name in ['model.safetensors.index.json',
                                     'pytorch_model.bin.index.json']:
                        idx_path = os.path.join(model_dir, idx_name)
                        if os.path.exists(idx_path):
                            with open(idx_path) as f:
                                idx = json.load(f)
                            shard_file = idx['weight_map'].get('score.bias')
                            if shard_file:
                                shard_path = os.path.join(model_dir, shard_file)
                                if shard_file.endswith('.safetensors'):
                                    from safetensors.torch import load_file
                                    bias_val = load_file(shard_path)['score.bias']
                                else:
                                    bias_val = torch.load(
                                        shard_path, map_location='cpu',
                                        weights_only=True)['score.bias']
                            break
                except Exception:
                    pass

            if bias_val is not None:
                # Replace score with a bias=True Linear
                old_w = score.weight.data
                new_score = nn.Linear(
                    old_w.shape[1], old_w.shape[0], bias=True,
                    device=old_w.device, dtype=old_w.dtype)
                new_score.weight = score.weight
                new_score.bias = nn.Parameter(
                    bias_val.to(device=old_w.device, dtype=old_w.dtype))
                model.score = new_score
                logger.info('SteerLMRMPlugin: patched score layer with bias from checkpoint.')

    def _build_steerlm_input(self, infer_request) -> str:
        """
        Construct the input text for the SteerLM RM.

        Per the HelpSteer2 paper (Sec 3): the RM scores the hidden state at the
        "end-of-response token", i.e. the final token of the assistant response.
        No label placeholder is needed — just the plain conversation.
        """
        BOS = '<|begin_of_text|>'
        SH  = '<|start_header_id|>'
        EH  = '<|end_header_id|>'
        EOT = '<|eot_id|>'

        parts = [BOS]
        parts.append(f'{SH}system{EH}\n\n{EOT}')

        messages = infer_request.get('messages', [])
        for msg in messages:
            role    = msg.get('role', '')
            content = msg.get('content', '') or ''
            if role == 'system':
                parts[-1] = f'{SH}system{EH}\n\n{content}{EOT}'
            elif role == 'user':
                parts.append(f'{SH}user{EH}\n\n{content}{EOT}')
            elif role == 'assistant':
                parts.append(f'{SH}assistant{EH}\n\n{content}{EOT}')

        return ''.join(parts)

    def __call__(self, inputs, **kwargs):
        tok = self.template.tokenizer
        device = self.model.device

        # Llama3 tokenizer has no pad_token by default.
        # MUST NOT use eos_token_id (<|eot_id|>=128009) as pad because our sequences
        # end with <|eot_id|> — LlamaForSequenceClassification would then treat the
        # real end token as padding and score the wrong position.
        # Use <|finetune_right_pad_id|>=128004 which never appears in normal text.
        if tok.pad_token_id is None:
            tok.pad_token_id = 128004  # <|finetune_right_pad_id|>
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = 128004

        # Build raw text, tokenize manually (bypass template chat formatting)
        texts = [self._build_steerlm_input(inp) for inp in inputs]
        encoded = tok(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=4096,
        )
        # Remove token_type_ids if present (Llama does not support it)
        encoded.pop('token_type_ids', None)
        reward_inputs = to_device(encoded, device)

        with torch.inference_mode():
            logits = self.model(**reward_inputs).logits  # [B, 9]
            # bias patched in __init__; ensure same device as logits (multi-GPU safe)
            score = getattr(self.model, 'score', None)
            if score is not None and getattr(score, 'bias', None) is not None:
                logits = logits + score.bias.to(logits.device)
            selected = logits[:, self.dim_indices]        # [B, n_dims]
            rewards = selected.sum(dim=-1).cpu()          # [B]
        return rewards.tolist()


class ArmoRMPlugin(DefaultRMPlugin):
    """
    Plugin for ArmoRM-Llama3-8B-v0.1 (trust_remote_code=True).

    Returns the sum of helpsteer-helpfulness + correctness + coherence
    (multi_obj_rewards indices 0, 1, 2) as a single scalar.

    Use armo_helpfulness / armo_correctness / armo_coherence /
    armo_complexity / armo_verbosity for per-dimension rewards with
    adaptive weighting. The trainer's reward model deduplication (rlhf.py)
    ensures the same checkpoint is only loaded once even when multiple
    plugin names are specified.

    multi_obj_rewards[:5] are HelpSteer2 attributes in order:
      0: helpsteer-helpfulness
      1: helpsteer-correctness
      2: helpsteer-coherence
      3: helpsteer-complexity
      4: helpsteer-verbosity
    """

    DIM_INDICES = [0, 1, 2]

    # Cache the latest inputs object to reuse one ArmoRM forward pass across
    # multiple per-dimension plugins in the same reward computation round.
    _cached_model = None
    _cached_inputs = None
    _cached_rewards = None

    def _score_batch(self, inputs, tok, device):
        """Batch inference, cached only while the same inputs object is reused."""
        if ArmoRMPlugin._cached_model is not self.model or ArmoRMPlugin._cached_inputs is not inputs:
            texts = [
                tok.apply_chat_template(inp.get('messages', []), tokenize=False)
                for inp in inputs
            ]
            encoded = tok(texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=4096).to(device)
            with torch.inference_mode():
                rewards = self.model(**encoded).rewards.float()  # [B, 19]
            ArmoRMPlugin._cached_model = self.model
            ArmoRMPlugin._cached_inputs = inputs
            ArmoRMPlugin._cached_rewards = rewards
        return ArmoRMPlugin._cached_rewards

    def __call__(self, inputs, **kwargs):
        tok = self.template.tokenizer
        device = self.model.device
        all_rewards = self._score_batch(inputs, tok, device)  # [B, 19]
        scores = all_rewards[:, self.DIM_INDICES].sum(dim=1)
        return scores.cpu()


class _ArmoRMSingleDimPlugin(ArmoRMPlugin):
    DIM = 0

    def __call__(self, inputs, **kwargs):
        tok = self.template.tokenizer
        device = self.model.device
        all_rewards = self._score_batch(inputs, tok, device)  # [B, 19]
        return all_rewards[:, self.DIM].cpu()


class ArmoRMHelpfulnessPlugin(_ArmoRMSingleDimPlugin):
    DIM = 0  # helpsteer-helpfulness


class ArmoRMCorrectnessPlugin(_ArmoRMSingleDimPlugin):
    DIM = 1  # helpsteer-correctness


class ArmoRMCoherencePlugin(_ArmoRMSingleDimPlugin):
    DIM = 2  # helpsteer-coherence


class ArmoRMComplexityPlugin(_ArmoRMSingleDimPlugin):
    DIM = 3  # helpsteer-complexity


class ArmoRMVerbosityPlugin(_ArmoRMSingleDimPlugin):
    DIM = 4  # helpsteer-verbosity


rm_plugins = {
    'default': DefaultRMPlugin,
    'genrm': GenRMPlugin,
    'steerlm': SteerLMRMPlugin,
    'armo': ArmoRMPlugin,
    'armo_helpfulness': ArmoRMHelpfulnessPlugin,
    'armo_correctness': ArmoRMCorrectnessPlugin,
    'armo_coherence': ArmoRMCoherencePlugin,
    'armo_complexity': ArmoRMComplexityPlugin,
    'armo_verbosity': ArmoRMVerbosityPlugin,
}
