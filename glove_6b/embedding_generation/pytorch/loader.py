# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GloVe-6B model loader implementation for word embedding generation.

Uses the NeuML/glove-6B StaticVectors model which provides 300-dimensional
GloVe word embeddings trained on 6 billion tokens.
"""
import torch
import torch.nn as nn
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GloVe-6B model variants for embedding generation."""

    GLOVE_6B = "NeuML/glove-6B"


class GloVeEmbeddingModel(nn.Module):
    """PyTorch nn.Embedding wrapper around StaticVectors GloVe weights."""

    def __init__(self, sv):
        super().__init__()
        import numpy as np

        weights = torch.from_numpy(np.array(sv.vectors, dtype="float32"))
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        norms = (emb * emb).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        return emb / norms


class ModelLoader(ForgeModel):
    """GloVe-6B model loader implementation for word embedding generation."""

    _VARIANTS = {
        ModelVariant.GLOVE_6B: ModelConfig(
            pretrained_model_name="NeuML/glove-6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLOVE_6B

    sample_sentences = ["This is an example sentence for generating word embeddings"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._sv = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GloVe-6B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from staticvectors import StaticVectors

        sv = StaticVectors(self._variant_config.pretrained_model_name)
        self._sv = sv
        model = GloVeEmbeddingModel(sv)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        sv = self._sv
        words = self.sample_sentences[0].split()
        token_ids = []
        for word in words:
            word_lower = word.lower()
            if word_lower in sv.tokens:
                token_ids.append(sv.tokens[word_lower])
            elif word in sv.tokens:
                token_ids.append(sv.tokens[word])
            else:
                token_ids.append(0)
        return torch.tensor([token_ids], dtype=torch.long)
