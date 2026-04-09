# TDR-001: LambdaRank vs Classification for Document Ranking

## Status: Accepted

## Context

Our binary relevance classifier (logistic regression / SVM) achieves NDCG@10 = 0.74 on the held-out evaluation set. While this is an improvement over BM25 (0.61), the classifier treats all relevant documents equally — a document at rank 1 and rank 10 receive the same gradient signal during training, even though rank 1 is 2.5x more valuable by NDCG's discount formula.

The question: should we switch from a classification objective to a ranking objective that directly optimizes for position-aware metrics?

## Decision

Switch to **LambdaRank** — a pairwise learning-to-rank approach that weights gradient updates by the NDCG impact of each document swap.

## Rationale

### Why position matters

NDCG applies a logarithmic discount to each position:
- Rank 1: weight 1.00
- Rank 2: weight 0.63
- Rank 3: weight 0.50
- Rank 5: weight 0.39
- Rank 10: weight 0.30

A classification loss treats all positions equally. A swap between ranks 1 and 2 gets the same gradient as a swap between ranks 98 and 99. LambdaRank fixes this by computing:

```
λ_ij = σ(-s_ij) × |ΔNDCG_ij|
```

where `s_ij = score_i - score_j` is the score difference between documents i and j, and `|ΔNDCG_ij|` is the absolute change in NDCG if we swap their positions.

This means the model focuses its learning capacity on swaps that actually move the metric — getting the top positions right.

### Implementation

Two-layer neural network:
- Input: 512 dims (TF-IDF + BM25 + metadata features), later expanded to 640 with ALS factors
- Hidden layers: 256 and 128 with ReLU and 10% dropout
- Output: 1 scalar relevance score

Training:
- Pairwise sampling within each query group
- Lambda gradients weighted by |ΔNDCG| 
- Gradient clipping at 1.0 (noisy relevance labels produce noisy lambdas)
- Cosine LR schedule over 20 epochs

## Alternatives Rejected

### Pointwise regression
Predicting absolute relevance scores (0-3) ignores relative ordering between documents for the same query. Two documents scored 2.1 and 2.0 may need to swap, but the gradient signal is tiny despite the swap potentially having large NDCG impact.

### ListNet (softmax over full list)
ListNet computes a softmax probability distribution over all documents per query and minimizes KL divergence to the ideal distribution. This is O(N²) per query at training time — at 100K documents per query (after retrieval candidates are returned), this becomes prohibitively expensive. Will be a concern revisited in Part 2's scale analysis.

### Direct NDCG optimization
NDCG involves sorting and position-based discounts, making it non-differentiable. Approximations exist (e.g., SoftNDCG) but introduce hyperparameters and approximation error. LambdaRank achieves the same goal through the lambda gradient trick without approximation.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Noisy relevance labels → noisy lambda gradients | Label smoothing, gradient clipping at 1.0 |
| Quadratic pair computation per query | Subsample pairs; focus on top-k documents |
| Overfitting on small query sets | Dropout, early stopping on validation NDCG |

## Results

- NDCG@10 = **0.83** after 20 epochs (up from 0.74 with SVM)
- Training curve plateaus at epoch 18
- Plateau diagnosis: the model has learned better weights over the same TF-IDF features, but the representation ceiling remains. "Cardiotoxicity" still doesn't help retrieve "cardiac adverse events."
- This motivates: (a) ALS factors for behavioral signals (Phase 5), (b) multimodal embeddings for semantic signals (Phase 6)
