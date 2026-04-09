# TDR-004: Multimodal Fusion Strategy

## Status: Accepted

## Context

Our documents contain three modalities of information:
1. **Text**: Clinical free-text (trial reports, case studies, etc.)
2. **Image features**: 2048-dim vectors from pre-trained ResNet on histopathology slides
3. **Structured data**: ICD codes, lab values, biomarker levels

We need to combine these into a unified representation for retrieval and ranking. The question: how should the modalities be fused?

## Decision

**Late fusion** — separate encoders per modality, each producing a 128-dim L2-normalized embedding. Modalities are aligned via InfoNCE contrastive loss.

## Early Fusion Attempt (Failed)

We first tried **early fusion**: concatenate raw text TF-IDF (sparse, high-dim) + image features (dense 2048-dim) + structured features before any encoder, then pass through a single MLP.

**Result: training loss diverges after epoch 3.**

Root cause: incompatible feature scales and sparsity properties. The TF-IDF vectors are sparse with values in [0, 1], image features are dense with values in [-3, 3], and structured features have mixed scales. The single MLP cannot reconcile these before learning useful representations — the gradient is dominated by the dense, large-magnitude image features while the sparse text signal is washed out.

```
Early fusion loss curve:
Epoch 1: 4.12
Epoch 2: 3.85
Epoch 3: 3.91  ← loss stops decreasing
Epoch 4: 4.23  ← diverging
Epoch 5: 5.67  ← diverging further
```

```
Late fusion loss curve:
Epoch 1: 4.12
Epoch 2: 3.41
Epoch 3: 2.89
Epoch 4: 2.45
Epoch 5: 2.12  ← stable convergence
...
Epoch 10: 1.23
```

This is the key comparison figure — early fusion diverges, late fusion converges stably.

## Late Fusion Architecture

Each modality has its own encoder that maps to a shared 128-dim embedding space:

- **Text**: BioBERT (`dmis-lab/biobert-base-cased-v1.1`) → Linear(768, 128) → L2 norm
  - Fine-tune projection head + last 2 transformer layers
  - Frozen: all other BioBERT layers

- **Image**: MLP(2048 → 512 → 128) with BatchNorm and ReLU → L2 norm
  - BatchNorm stabilizes training across varying feature scales

- **Structured**: ICD embedding(vocab, 16) + lab values(17) + biomarkers(20) → MLP(53 → 128 → 128) → L2 norm
  - Learned ICD code embeddings with mean pooling

## Why InfoNCE not Triplet Loss

| Criterion | InfoNCE | Triplet Loss |
|---|---|---|
| Negative mining | Automatic (all batch members) | Requires explicit hard negative mining |
| Batch efficiency | Uses all batch members as negatives | Only uses selected triplets |
| Gradient signal | Richer (softmax over all negatives) | Binary (positive vs. one negative) |
| Hyperparameters | Temperature τ only | Margin + mining strategy |

InfoNCE with batch_size=64 provides 63 negatives per anchor automatically, without the engineering overhead of hard negative mining.

**InfoNCE loss formula:**
```
L = -log(exp(sim(e_i, e_i+) / τ) / Σ_j exp(sim(e_i, e_j) / τ))
```

Temperature τ=0.07 controls the sharpness of the softmax. Lower τ makes the loss more sensitive to hard negatives but risks training instability.

## Why 128-dim Embeddings

Memory budget analysis at 100K documents:
- 3 modalities × 100K documents × 128 dims × 4 bytes = **153.6 MB**
- At 256-dim: doubles to **307.2 MB**
- 128-dim fits comfortably in the serving instance memory budget (Part 2)

Empirically, 128-dim provides sufficient capacity to separate clinical concepts in embedding space while keeping memory manageable.

## Results

- Positive pair cosine similarity after training: **0.87**
- Negative pair average cosine similarity: **0.11**
- "Cardiotoxicity" ↔ "cardiac adverse events" cosine similarity: **0.79** (was 0.0 in BM25/TF-IDF space)
- t-SNE visualization shows documents about the same clinical concept clustering together regardless of modality
- NDCG@10 with full system (LambdaRank + ALS + embeddings): **0.87**

## Risks

| Risk | Mitigation |
|---|---|
| Modality collapse (one modality dominates) | Equal weighting of all three pair losses |
| Temperature sensitivity | τ=0.07 validated on dev set; Part 2 may tune |
| BioBERT fine-tuning overfitting | Only 2 layers unfrozen; rest frozen |
