"""
ref: https://github.com/aisolab/nlp_implementation/blob/master/Character-level_Convolutional_Networks_for_Text_Classification/utils.py
"""

from __future__ import absolute_import
import torch
from pathlib import Path


class CheckpointManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir

    def save_checkpoint(self, state, filename):
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename):
        state = torch.load(self._model_dir / filename, map_location=torch.device('cpu'))
        return state