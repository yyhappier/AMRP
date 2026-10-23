# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Vicuna template for text-only models
Based on the official Vicuna conversation format
"""

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template

# Vicuna default system message
VICUNA_DEFAULT_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Register Vicuna template
# Format:
# A chat between a curious user and an artificial intelligence assistant...
#
# USER: {query}
# ASSISTANT: {response}
register_template(
    TemplateMeta(
        LLMTemplateType.vicuna,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}}\nASSISTANT:'],
        chat_sep=['\n'],
        suffix=['</s>'],
        default_system=VICUNA_DEFAULT_SYSTEM,
        system_prefix=['<s>{{SYSTEM}}\n\n']
    )
)
