# Post-Mortem Log

A living record of failures, pivots, and lessons learned. Every entry documents what broke, why it broke, and what we did about it. This is not a clean retrospective — it's a research log written as things happened.

---

## PM-001: BM25 Vocabulary Mismatch (Week 2)

**What happened**: BM25 baseline achieved NDCG@10 = 0.61. Investigated failure cases and found that clinically synonymous terms score zero against each other. "Cardiotoxicity" and "cardiac adverse events" share zero tokens.

**Root cause**: BM25 is a term-matching algorithm. It cannot understand that different words mean the same thing. This is a fundamental limitation of the representation, not a tuning problem.

**What we tried**: Adjusting k1 (1.0-2.5) and b (0.5-1.0) parameters. Maximum NDCG@10 achieved: 0.63. The 0.02 improvement confirms this is not a parameter sensitivity issue.

**What we learned**: The bottleneck is vocabulary mismatch, not ranking weakness. No amount of BM25 tuning addresses this. Learned representations are needed.

**Impact**: Motivated the entire ML pipeline that followed.

---

## PM-002: TF-IDF Feature Ceiling (Week 4)

**What happened**: Logistic regression (NDCG@10 = 0.71) and SVM (0.74) both plateau despite adding more features. Ablation study shows diminishing returns after 6 features.

**Root cause**: TF-IDF features encode term presence/frequency but not meaning. "Cardiotoxicity" and "cardiac adverse events" remain orthogonal dimensions in TF-IDF space regardless of how many features we derive from them.

**What we learned**: Model sophistication cannot compensate for inadequate feature representation. This is a ceiling, not a floor — no model can surpass it without better features.

**Impact**: Motivated two parallel tracks: (a) LambdaRank for better ranking over existing features, (b) learned embeddings for better features.

---

## PM-003: Early Fusion Divergence (Week 8)

**What happened**: Attempted early fusion for multimodal embeddings — concatenate raw TF-IDF (sparse, high-dim) + image features (dense 2048-dim) + structured features before any encoder. Training loss diverged after epoch 3.

**Root cause**: Incompatible feature scales and sparsity properties. TF-IDF vectors are sparse with values in [0, 1]. Image features are dense with values in [-3, 3]. The gradient is dominated by dense, large-magnitude image features while the sparse text signal is washed out.

**Loss curve**:
```
Epoch 1: 4.12
Epoch 2: 3.85
Epoch 3: 3.91 ← stops decreasing
Epoch 4: 4.23 ← diverging
Epoch 5: 5.67 ← worse
```

**What we did**: Switched to late fusion — separate encoders per modality, meeting at the 128-dim embedding level. Loss converged stably.

**What we learned**: Raw feature concatenation across modalities with different statistical properties is fragile. Each modality needs its own encoder to map to a compatible representation before fusion.

**Impact**: Late fusion became the architecture (TDR-004).

---

## PM-004: ALS Cold Start at Scale (Week 10)

**What happened**: ALS collaborative filtering improved NDCG@10 from 0.83 to 0.85 — a modest gain. Investigation revealed 40% of documents at 10K scale have zero interactions.

**Root cause**: Power-law distribution of interactions. Popular documents (top 20%) attract 60% of all interactions. The long tail of documents has zero or minimal interaction data, producing zero ALS factor vectors.

**What we learned**: Collaborative filtering helps for well-interacted documents but provides zero signal for the majority of the corpus. The cold start problem is inherent to behavioral-signal-based methods and cannot be solved within CF alone.

**Mitigation**: Content-based features (TF-IDF, then embeddings) serve as the fallback. The combined system uses ALS factors when available and degrades gracefully to content features for cold start documents.

**Impact**: Confirmed that we need content-understanding capabilities (embeddings) alongside behavioral signals (CF) — neither alone is sufficient.

---

## PM-005: Sparsity Wall at 100K (Week 10)

**What happened**: Interaction matrix sparsity at 100K documents × 500 physicians = 50M possible pairs with only ~15K observed interactions = 99.97% sparse. ALS convergence slowed dramatically compared to 1K scale.

**Root cause**: With 99.97% sparsity, the vast majority of the factor matrix is informed by the regularization term rather than actual data. The model is essentially guessing for most document-user pairs.

**Memory note**: Item factor matrix alone = 100K × 128 × 4 bytes = 51.2 MB. Combined with TF-IDF vectors and training data, total RAM approaches 4 GB. This will be a constraint in Part 2's serving architecture.

**What we learned**: CF-based methods have a practical sparsity limit. Beyond ~99% sparsity, the signal-to-noise ratio in the factor matrices degrades significantly.

**Impact**: Foreshadows the scale challenges addressed in Part 2.

---

## Lessons Learned (So Far)

1. **Always start simple**: BM25 baseline took 2 days. It revealed the core problem (vocabulary mismatch) that shaped the entire project.

2. **Ablation before escalation**: The TF-IDF ablation study (PM-002) saved us from trying deeper models on the same features. The ceiling was in the representation, not the model.

3. **Document failures explicitly**: PM-003 (early fusion divergence) would have been invisible if we'd just switched to late fusion without recording why.

4. **Cold start is universal**: Any method that relies on interaction data will struggle with the long tail. Content-based fallbacks are not optional.

5. **Sparsity compounds**: 85% → 97% → 99.97% sparsity across 10x scale jumps. Methods that work at 1K may not work at 100K. Test at target scale early.
