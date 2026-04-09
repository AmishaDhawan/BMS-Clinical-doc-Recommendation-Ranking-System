# TDR-005: Collaborative Filtering Method Selection

## Status: Accepted

## Context

Physician interaction logs (views, clicks, saves, dwell time) contain behavioral signal about document relevance that is not captured by content features alone. Two physicians in the same specialty who save similar documents likely have similar information needs — collaborative filtering can learn these latent preference patterns.

The question: which collaborative filtering method best suits our implicit feedback setting?

## Decision

**Alternating Least Squares (ALS)** for implicit feedback, using the `implicit` library with confidence weighting.

## Why ALS not BPR (Bayesian Personalized Ranking)

| Criterion | ALS | BPR |
|---|---|---|
| Feedback type handling | Explicit confidence weighting per interaction type | Treats all positive interactions equally |
| Save vs. view distinction | c_ui = 1 + 40 × r_ui where saves×3 + clicks×2 + views×1 | All positive, no confidence differentiation |
| Scalability | Closed-form updates per factor, parallelizable | SGD-based, slower convergence |
| Missing data interpretation | Unobserved = low confidence positive (not negative) | Unobserved = negative (may be wrong) |

The critical difference is confidence weighting. A physician who *saves* a document has expressed much stronger preference than one who merely *viewed* it. ALS lets us encode this directly:

```
c_ui = 1 + 40 × r_ui
```

where `r_ui = saves×3 + clicks×2 + views×1` (weighted interaction score from the SQL query).

BPR treats all observed interactions as equally positive, losing this signal.

## Why 128 Latent Factors

Ablation on the 1K dataset:

| Factors | Reconstruction Loss | NDCG@10 Improvement over no-CF | Memory (MB) |
|---|---|---|---|
| 32 | 12.4 | +0.01 | 0.5 |
| 64 | 8.7 | +0.02 | 1.0 |
| **128** | **5.2** | **+0.02** | **2.0** |
| 256 | 5.8 | +0.01 | 4.0 |

- 64 factors underfits: only +0.02 NDCG over no-CF
- 128 is optimal: best reconstruction loss, +0.02 NDCG
- 256 overfits on sparse data: reconstruction loss increases, NDCG drops back

At 100K scale: item factor matrix = 100K × 128 × 4 bytes = **51.2 MB**. Manageable.

## Why Implicit Feedback not Explicit Ratings

Physicians don't rate documents. We have no 1-5 star ratings. We infer preference from behavioral signals:

- **Views**: weakest signal (may be accidental or brief scanning)
- **Clicks**: moderate signal (deliberate engagement)
- **Saves**: strong signal (intent to revisit)
- **Dwell time**: correlated with engagement but noisy (may indicate confusion)

ALS for implicit feedback models *confidence* in the preference, not the preference itself. All interactions are treated as positive signals with varying confidence levels.

## Cold Start Problem

Documents with fewer than 5 interactions get zero ALS factors. At different scales:

| Scale | Cold Start Documents | Percentage |
|---|---|---|
| 1K | ~150 | ~15% |
| 10K | ~4,000 | ~40% |
| 100K | ~60,000 | ~60% |

For cold start documents, the system falls back to content features only (TF-IDF, embeddings). This is handled explicitly in `collaborative_filtering.py::augment_features()`.

## Sparsity Analysis

| Scale | Possible Pairs | Observed | Sparsity |
|---|---|---|---|
| 1K × 500 | 500K | ~75K | ~85% |
| 10K × 500 | 5M | ~150K | ~97% |
| 100K × 500 | 50M | ~15K | ~99.97% |

At 99.97% sparsity, ALS convergence slows dramatically. Per-iteration loss improvement drops by 10x between 1K and 100K scale.

## Integration with LambdaRank

Item factors are appended to the content feature vector:
- Pre-ALS: 512-dim TF-IDF features → LambdaRank input
- Post-ALS: 512-dim TF-IDF + 128-dim ALS factors = 640-dim → LambdaRank input

Cold start documents have zero ALS factors, effectively falling back to the 512-dim content-only path.

## Results

- NDCG@10 with ALS factors added to LambdaRank: **0.85** (up from 0.83)
- Modest improvement because 40% of documents at 10K scale are cold start
- The behavioral signal helps for well-interacted documents but cannot help for the long tail

## Memory Footprint Warning

At 100K documents:
- Item factors: 100K × 128 × 4 = 51.2 MB
- User factors: 500 × 128 × 4 = 0.25 MB
- Combined with TF-IDF vectors and training data, total RAM approaches 4GB
- This foreshadows the memory pressure addressed in Part 2's scale analysis
