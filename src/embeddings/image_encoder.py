"""
Image encoder for 2048-dim synthetic image feature vectors.

Takes pre-extracted 2048-dim feature vectors (simulated ResNet output on
histopathology slides) and projects them to 128-dim embedding space via
a 2-layer MLP with BatchNorm and ReLU.

Architecture:
    2048 → Linear(2048, 512) → BatchNorm → ReLU → Linear(512, 128) → L2 normalize

These are NOT actual images — they simulate the output of a pre-trained ResNet
applied to histopathology slides. In production, an actual ResNet would extract
these features from slide images before this encoder processes them.

Phase 6 integration:
    The 128-dim output is aligned with text and structured embeddings via
    InfoNCE contrastive loss. Positive pairs are different modalities from
    the same document; negatives are all other documents in the batch.
"""

import numpy as np
import torch
import torch.nn as nn


class ImageFeatureEncoder(nn.Module):
    """MLP encoder for pre-extracted image feature vectors.

    Architecture:
        2048 → 512 (BatchNorm + ReLU) → 128 (L2 normalized)

    Why BatchNorm:
        The synthetic 2048-dim features have varying scales across dimensions.
        BatchNorm stabilizes training and speeds convergence. Without it,
        the contrastive loss takes 3x longer to converge.

    Why 2-layer not deeper:
        The input is already a learned representation (ResNet features).
        A deeper MLP risks overfitting on our synthetic data. 2 layers
        provide sufficient capacity for the linear projection to the
        shared embedding space.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image features to 128-dim normalized embedding.

        Args:
            x: Image feature vectors of shape (batch_size, 2048).

        Returns:
            L2-normalized embeddings of shape (batch_size, 128).
        """
        encoded = self.encoder(x)
        normalized = nn.functional.normalize(encoded, p=2, dim=1)
        return normalized

    def encode_features(
        self,
        features: np.ndarray,
        batch_size: int = 256,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Encode a batch of image feature vectors.

        Args:
            features: Feature matrix of shape (n_docs, 2048).
            batch_size: Batch size for encoding.
            device: Device for computation.

        Returns:
            Embedding matrix of shape (n_docs, 128).
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = torch.FloatTensor(features[i:i + batch_size]).to(device)
                embeddings = self.forward(batch)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
