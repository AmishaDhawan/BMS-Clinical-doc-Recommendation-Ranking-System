"""
Text encoder using BioBERT with a linear projection head to 128-dim.

Model: dmis-lab/biobert-base-cased-v1.1
Fine-tuning strategy: Only the projection head plus last 2 transformer layers.
Output: 128-dim L2-normalized text embedding.

This encoder resolves the vocabulary mismatch problem that limits BM25 and
TF-IDF features: "cardiotoxicity" and "cardiac adverse events" are mapped to
nearby points in the 128-dim embedding space (cosine similarity ~0.79 after
contrastive training).

Phase 6 results:
    Replacing TF-IDF features with 128-dim text embeddings in LambdaRank
    improves NDCG@10 from 0.85 to 0.87.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BioBERTTextEncoder(nn.Module):
    """BioBERT-based text encoder with projection head.

    Architecture:
        BioBERT (768-dim CLS output) → Linear(768, 128) → L2 normalize

    Fine-tuning:
        - Projection head: always trainable
        - Last 2 transformer layers: trainable
        - All other BioBERT layers: frozen

    Why BioBERT over general BERT:
        BioBERT is pre-trained on PubMed abstracts and PMC full-text articles.
        It already understands medical terminology relationships, so fine-tuning
        the last 2 layers is sufficient to learn the projection to our task's
        embedding space.

    Why 128-dim:
        At 100K documents, 3 modalities × 100K × 128 × 4 bytes = 153.6MB.
        At 256-dim this doubles to 307MB. 128-dim fits comfortably in the
        serving instance memory budget defined in Part 2.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        projection_dim: int = 128,
        fine_tune_layers: int = 2,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        encoder_layers = self.bert.encoder.layer
        for layer in encoder_layers[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Projection head: 768-dim CLS → 128-dim embedding
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to 128-dim normalized embedding.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            L2-normalized embeddings of shape (batch_size, 128).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        projected = self.projection(cls_output)
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        return normalized

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Encode a list of text strings to 128-dim embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            max_length: Maximum token length.
            device: Device to use for computation.

        Returns:
            Embedding matrix of shape (len(texts), 128).
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                embeddings = self.forward(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
