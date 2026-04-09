"""
Structured data encoder for clinical metadata.

Encodes structured features (ICD codes, lab values, biomarker levels) into
a 128-dim embedding via a 2-layer MLP.

Input features:
    - ICD code embeddings (learned embedding table)
    - Normalized lab values (min-max scaled)
    - Biomarker levels (min-max scaled)
    - Concatenated → MLP → 128-dim output

Architecture:
    concat(icd_embed, lab_values, biomarker_levels) → Linear → ReLU → Linear → L2 normalize

Phase 6 integration:
    The 128-dim output is aligned with text and image embeddings via InfoNCE
    contrastive loss. Structured features capture information not present in
    free text (exact lab values, coded diagnoses) that complements the
    semantic understanding from text embeddings.
"""

import numpy as np
import torch
import torch.nn as nn


# Fixed vocabulary of ICD codes used in the synthetic data
ICD_CODE_VOCAB = [
    "C34.1", "C50.9", "C61", "C18.9", "C71.9",
    "I25.1", "I50.9", "I42.0", "I48.0",
    "N17.9", "N18.9",
    "K72.0", "K75.9",
    "D70.9", "D69.6", "D64.9",
    "G62.0", "G93.4",
    "E11.9", "E05.0",
    "J84.1", "J96.0",
    "M35.9", "M06.9",
    "D89.9", "D80.9",
]

# Fixed vocabulary of lab tests
LAB_TEST_VOCAB = [
    "creatinine", "ALT", "AST", "bilirubin", "albumin",
    "hemoglobin", "platelet_count", "neutrophil_count",
    "TSH", "free_T4", "troponin", "BNP", "LDH",
    "CEA", "CA-125", "PSA", "AFP",
]

# Fixed vocabulary of biomarkers
BIOMARKER_VOCAB = [
    "PD-L1", "EGFR", "ALK", "KRAS", "BRAF_V600E", "HER2", "BRCA1/2",
    "MSI-H", "TMB", "NTRK", "ROS1", "MET", "RET", "PIK3CA",
    "TP53", "IDH1", "FGFR", "CDK4", "BCL-2", "BTK",
]


class StructuredEncoder(nn.Module):
    """MLP encoder for structured clinical metadata.

    Encodes ICD codes (via learned embeddings), lab values, and biomarker
    levels into a shared 128-dim embedding space.

    Architecture:
        ICD codes → Embedding(vocab_size, 16) → mean pool → 16-dim
        Lab values → normalized → 17-dim
        Biomarker levels → normalized → 20-dim
        Concat(16 + 17 + 20 = 53) → Linear(53, 128) → ReLU → Linear(128, 128) → L2 norm
    """

    def __init__(
        self,
        output_dim: int = 128,
        icd_embed_dim: int = 16,
    ):
        super().__init__()

        self.output_dim = output_dim

        # ICD code embedding
        self.icd_vocab = {code: i + 1 for i, code in enumerate(ICD_CODE_VOCAB)}
        self.icd_embedding = nn.Embedding(
            num_embeddings=len(ICD_CODE_VOCAB) + 1,  # +1 for padding
            embedding_dim=icd_embed_dim,
            padding_idx=0,
        )

        # Input dimension: icd_embed_dim + n_lab_tests + n_biomarkers
        input_dim = icd_embed_dim + len(LAB_TEST_VOCAB) + len(BIOMARKER_VOCAB)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def encode_icd_codes(self, icd_codes_batch: list[list[str]]) -> torch.Tensor:
        """Encode ICD codes via learned embeddings with mean pooling.

        Args:
            icd_codes_batch: List of ICD code lists, one per document.

        Returns:
            Mean-pooled ICD embeddings of shape (batch_size, icd_embed_dim).
        """
        device = self.icd_embedding.weight.device
        batch_size = len(icd_codes_batch)
        max_codes = max(len(codes) for codes in icd_codes_batch) if icd_codes_batch else 1

        # Pad and encode
        indices = torch.zeros(batch_size, max_codes, dtype=torch.long, device=device)
        for i, codes in enumerate(icd_codes_batch):
            for j, code in enumerate(codes):
                indices[i, j] = self.icd_vocab.get(code.strip(), 0)

        embeddings = self.icd_embedding(indices)  # (batch, max_codes, embed_dim)

        # Mean pool (ignoring padding)
        mask = (indices != 0).unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled

    def encode_lab_values(self, lab_values_batch: list[dict]) -> torch.Tensor:
        """Encode lab values as a fixed-length vector.

        Args:
            lab_values_batch: List of lab value dicts, one per document.

        Returns:
            Lab value vectors of shape (batch_size, n_lab_tests).
        """
        device = self.icd_embedding.weight.device
        batch_size = len(lab_values_batch)
        vectors = torch.zeros(batch_size, len(LAB_TEST_VOCAB), device=device)

        for i, lab_dict in enumerate(lab_values_batch):
            for j, test in enumerate(LAB_TEST_VOCAB):
                # Normalize lab test name for matching
                test_normalized = test.replace("_", " ")
                for key, value in lab_dict.items():
                    if key.replace("_", " ").lower() == test_normalized.lower():
                        vectors[i, j] = value / 15.0  # Simple normalization
                        break

        return vectors

    def encode_biomarkers(self, biomarker_batch: list[dict]) -> torch.Tensor:
        """Encode biomarker levels as a fixed-length vector.

        Args:
            biomarker_batch: List of biomarker dicts, one per document.

        Returns:
            Biomarker vectors of shape (batch_size, n_biomarkers).
        """
        device = self.icd_embedding.weight.device
        batch_size = len(biomarker_batch)
        vectors = torch.zeros(batch_size, len(BIOMARKER_VOCAB), device=device)

        for i, bm_dict in enumerate(biomarker_batch):
            for j, bm in enumerate(BIOMARKER_VOCAB):
                bm_normalized = bm.replace("_", " ")
                for key, value in bm_dict.items():
                    if key.replace("_", " ") == bm_normalized:
                        vectors[i, j] = value / 100.0  # Normalize to [0, 1]
                        break

        return vectors

    def forward(
        self,
        icd_codes: list[list[str]],
        lab_values: list[dict],
        biomarkers: list[dict],
    ) -> torch.Tensor:
        """Encode structured features to 128-dim normalized embedding.

        Args:
            icd_codes: List of ICD code lists per document.
            lab_values: List of lab value dicts per document.
            biomarkers: List of biomarker dicts per document.

        Returns:
            L2-normalized embeddings of shape (batch_size, 128).
        """
        icd_embed = self.encode_icd_codes(icd_codes)
        lab_vec = self.encode_lab_values(lab_values)
        bm_vec = self.encode_biomarkers(biomarkers)

        combined = torch.cat([icd_embed, lab_vec, bm_vec], dim=1)
        encoded = self.encoder(combined)
        normalized = nn.functional.normalize(encoded, p=2, dim=1)

        return normalized

    def forward_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pre-computed structured feature tensor.

        Used when structured features have already been converted to a
        fixed-length tensor (e.g., during batch training).

        Args:
            x: Structured feature tensor of shape (batch_size, input_dim).

        Returns:
            L2-normalized embeddings of shape (batch_size, 128).
        """
        encoded = self.encoder(x)
        normalized = nn.functional.normalize(encoded, p=2, dim=1)
        return normalized
