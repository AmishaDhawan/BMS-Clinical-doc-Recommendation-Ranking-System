# BMS Clinical Document Recommendation & Ranking System

## The Research Story

This repository documents a 6-month journey building a clinical document retrieval and ranking system for a pharmaceutical company. Physicians and clinical researchers retrieve medical documents from a corpus that grows from 1,000 to 100,000 records — clinical trial reports, drug interaction studies, pathology reports with histopathology image features, and patient cohort summaries with structured lab data.

The existing system was BM25 keyword search. It failed because medical language is inconsistent: **"cardiotoxicity" and "cardiac adverse events" are the same clinical concept but share zero tokens.** Physician search time was high. The research question: can we build a retrieval and ranking system that understands semantic relevance and serves at under 100ms p99 latency?

This is a living research log, not a clean retrospective. Every pivot is motivated by something that actually broke or underperformed. We climbed the complexity ladder one rung at a time.

---

## NDCG@10 Progression

| Phase | Approach | NDCG@10 | Key Learning |
|-------|----------|---------|--------------|
| 2 | BM25 keyword baseline | 0.61 | Vocabulary mismatch is the core problem |
| 3 | Logistic regression | 0.71 | Learns feature weights from labeled data |
| 3 | Linear SVM | 0.74 | Max-margin in high-dim TF-IDF space |
| 4 | LambdaRank (TF-IDF) | 0.83 | Position-aware NDCG optimization |
| 5 | LambdaRank + ALS | 0.85 | Behavioral signals help warm documents |
| 6 | Full system (multimodal) | **0.87** | Contrastive embeddings resolve vocabulary mismatch |

---

## Project Structure

```
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── schema.sql                          # SQLite schema (5 tables)
│   └── synthetic/                          # Generated datasets (1K/10K/100K)
├── src/
│   ├── data_pipeline/
│   │   ├── generator.py                    # Synthetic data with vocabulary mismatch
│   │   ├── sql_queries.py                  # Production SQL (CTEs, window functions)
│   │   └── loader.py                       # Data loading via SQL queries
│   ├── retrieval/
│   │   └── bm25.py                         # BM25 from scratch (not wrapped)
│   ├── ranking/
│   │   ├── logistic_baseline.py            # Logistic regression on TF-IDF
│   │   ├── svm_baseline.py                 # Linear SVM on TF-IDF
│   │   ├── lambdarank.py                   # LambdaRank from scratch in PyTorch
│   │   └── collaborative_filtering.py      # ALS with implicit feedback
│   ├── embeddings/
│   │   ├── text_encoder.py                 # BioBERT + projection to 128-dim
│   │   ├── image_encoder.py                # MLP: 2048 → 512 → 128
│   │   ├── structured_encoder.py           # ICD codes + lab values → 128-dim
│   │   └── contrastive_trainer.py          # InfoNCE loss training
│   ├── serving/                            # (Part 2)
│   └── evaluation/                         # (Part 2)
├── notebooks/
│   ├── 01_problem_exploration.ipynb        # Data distributions, sparsity, mismatch
│   ├── 02_keyword_baseline.ipynb           # BM25 failure analysis
│   ├── 03_ml_baselines.ipynb               # Feature ceiling diagnosis
│   ├── 04_lambdarank.ipynb                 # Neural ranking with NDCG optimization
│   ├── 05_collaborative_filtering.ipynb    # ALS cold start analysis
│   └── 06_multimodal_embeddings.ipynb      # Contrastive embeddings, t-SNE
├── docs/
│   ├── problem_statement.md
│   ├── post_mortem_log.md                  # 5 documented failures
│   └── tdr/
│       ├── TDR-001-lambdarank-vs-classification.md
│       ├── TDR-003-faiss-index-selection.md
│       ├── TDR-004-multimodal-fusion-strategy.md
│       └── TDR-005-collaborative-filtering-method.md
├── configs/
│   ├── 1k_config.yaml
│   ├── 10k_config.yaml
│   └── 100k_config.yaml
└── logs/
    └── training/
```

---

## Phase-by-Phase Summary

### Phase 1: Data Pipeline and Problem Characterization
Built a synthetic data generator producing clinical documents with realistic free-text, 2048-dim image feature vectors (simulated ResNet output), and structured metadata. Three dataset sizes (1K, 10K, 100K) with 500 synthetic physician users.

**Key observation**: Interaction matrix sparsity increases from ~85% (1K) to ~97% (10K) to **~99.97% (100K)**. This foreshadows collaborative filtering struggling at scale.

### Phase 2: BM25 Baseline — Document the Failure
Implemented BM25 from scratch. NDCG@10 = 0.61.

**Root cause**: Vocabulary mismatch. "Cardiotoxicity" and "cardiac adverse events" share zero tokens. No amount of BM25 tuning addresses this. The problem is representation, not ranking.

**Latency**: Linear degradation — 2ms p99 at 1K, 18ms at 10K, 180ms at 100K.

### Phase 3: ML Baselines — Find the Feature Ceiling
Logistic regression (NDCG@10 = 0.71) and SVM (0.74) on handcrafted TF-IDF features.

**Critical diagnosis**: Adding more TF-IDF features doesn't improve beyond ~0.75. The bottleneck is feature representation, not model complexity. This motivates (a) a ranking objective instead of classification, and (b) learned semantic embeddings.

### Phase 4: LambdaRank — Neural Ranking
Implemented LambdaRank from scratch in PyTorch. Two-layer network (512→256→128→1) with lambda gradients weighted by |ΔNDCG|.

NDCG@10 = **0.83** — a +0.09 jump from SVM. Training plateaus at epoch 18: the model has squeezed everything possible from TF-IDF features.

See [TDR-001](docs/tdr/TDR-001-lambdarank-vs-classification.md) for the architectural decision.

### Phase 5: Collaborative Filtering — ALS for Implicit Feedback
ALS with 128 latent factors, confidence weighting c_ui = 1 + 40×r_ui. Appended 128-dim ALS factors to TF-IDF: 512 + 128 = **640-dim combined input**.

NDCG@10 = 0.85 (+0.02). Modest because **40% of documents at 10K are cold start** (< 5 interactions).

See [TDR-005](docs/tdr/TDR-005-collaborative-filtering-method.md) for the method choice.

### Phase 6: Multimodal Contrastive Embeddings
Three modality-specific encoders (text via BioBERT, image MLP, structured MLP) projecting to 128-dim. Trained with InfoNCE contrastive loss (temperature τ=0.07).

**Early fusion failed** — training loss diverged after epoch 3 due to incompatible feature scales. Switched to late fusion.

After training: "cardiotoxicity" ↔ "cardiac adverse events" cosine similarity = **0.79** (was 0.0 in TF-IDF). Full system NDCG@10 = **0.87**.

See [TDR-004](docs/tdr/TDR-004-multimodal-fusion-strategy.md) for the fusion strategy decision.

---

## Key Architectural Decisions (Part 2 depends on these)

| Decision | Value | Rationale |
|----------|-------|-----------|
| Embedding dimension | 128-dim per modality | 153.6MB total at 100K (3 modalities). 256-dim doubles to 307MB. |
| LambdaRank input dim | 640-dim | 512 TF-IDF + 128 ALS factors |
| ALS latent factors | 128 | Ablation: 64 underfits, 256 overfits on sparse data |
| Confidence weight | α=40, c_ui = 1 + 40×r_ui | Saves (×3) > clicks (×2) > views (×1) |
| Contrastive temperature | τ=0.07 | Standard for InfoNCE |
| BioBERT model | dmis-lab/biobert-base-cased-v1.1 | Domain-specific pre-training |
| Image features | 2048-dim (simulated ResNet) | Not actual images — synthetic feature vectors |

---

## Documented Failures

1. **PM-001**: BM25 vocabulary mismatch (NDCG@10 = 0.61)
2. **PM-002**: TF-IDF feature ceiling (0.74 max despite more features)
3. **PM-003**: Early fusion divergence (loss diverged after epoch 3)
4. **PM-004**: ALS cold start at scale (40% of docs at 10K have zero interactions)
5. **PM-005**: Sparsity wall at 100K (99.97% sparse, ALS convergence slows)

See [post_mortem_log.md](docs/post_mortem_log.md) for detailed root cause analysis.

---

## Setup

```bash
pip install -r requirements.txt
```

Generate synthetic data:
```python
from src.data_pipeline.generator import generate_dataset
db_path = generate_dataset("configs/1k_config.yaml", seed=42)
```

---

## What's Next (Part 2)

Scale analysis, inference optimization, serving architecture, and full evaluation suite (including A/B test design and causal inference framing) are covered in the Part 2 continuation of this project, currently in development.
