# Copyright (c) Alibaba, Inc. and its affiliates.
"""
HelpSteer2 reward functions for GRPO training.

Uses the three human-annotated attributes from the HelpSteer2 dataset as reward signals:
  - helpfulness : overall helpfulness of the response (0-4)
  - correctness : factual accuracy and completeness (0-4)
  - coherence   : consistency and clarity of expression (0-4)

Each ORM reads the corresponding field from the dataset row directly (no live RM inference).
The combined reward is the sum of the three dimensions (scale 0-12), or the caller can
use --reward_weights to scale each dimension independently.

Data fields expected in each training sample:
  helpfulness : int, 0-4
  correctness : int, 0-4
  coherence   : int, 0-4

These fields must be present in the GRPO dataset JSONL (produced by
scripts/data/convert_helpsteer2_to_swift.py which copies them verbatim from HelpSteer2).
"""

from typing import List, Optional


class HelpSteer2HelpfulnessReward:
    """Returns the helpfulness label (0-4) normalised to [0, 1]."""

    def __call__(self, completions: List[str], **kwargs) -> List[Optional[float]]:
        labels = kwargs.get('helpfulness')
        if labels is None:
            return [None] * len(completions)
        return [float(v) / 4.0 if v is not None else None for v in labels]


class HelpSteer2CorrectnessReward:
    """Returns the correctness label (0-4) normalised to [0, 1]."""

    def __call__(self, completions: List[str], **kwargs) -> List[Optional[float]]:
        labels = kwargs.get('correctness')
        if labels is None:
            return [None] * len(completions)
        return [float(v) / 4.0 if v is not None else None for v in labels]


class HelpSteer2CoherenceReward:
    """Returns the coherence label (0-4) normalised to [0, 1]."""

    def __call__(self, completions: List[str], **kwargs) -> List[Optional[float]]:
        labels = kwargs.get('coherence')
        if labels is None:
            return [None] * len(completions)
        return [float(v) / 4.0 if v is not None else None for v in labels]
