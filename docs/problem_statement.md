# BMS Clinical Document Recommendation & Ranking System

## Problem Statement

### Context

Physicians and clinical researchers at a pharmaceutical company retrieve medical documents from a corpus that grows from 1,000 to 100,000 records. Documents include:

- **Clinical trial reports**: Phase I-III studies with efficacy and safety endpoints
- **Drug interaction studies**: Pharmacokinetic and pharmacodynamic interaction analyses
- **Pathology reports**: Histopathology findings with image feature vectors (2048-dim simulated ResNet output)
- **Patient cohort summaries**: Retrospective analyses with structured lab data and biomarker levels
- **Case studies**: Individual patient case reports with treatment outcomes

### The Existing System

The current retrieval system is **BM25 keyword search**. It works by matching query terms to document terms using term frequency and inverse document frequency weighting.

### Why It Fails

Medical language is inherently inconsistent. The same clinical concept is expressed with completely different vocabulary across documents:

| Clinical Concept | Term Variant 1 | Term Variant 2 | Term Variant 3 |
|---|---|---|---|
| Heart damage from drugs | cardiotoxicity | cardiac adverse events | TKI-induced cardiac events |
| Liver damage | hepatotoxicity | drug-induced liver injury | transaminase elevation |
| Immune reaction | immunogenicity | autoimmune activation | cytokine storm |
| Kidney injury | nephrotoxicity | acute kidney injury | creatinine elevation |
| Blood cell suppression | myelosuppression | pancytopenia | bone marrow suppression |

These synonym groups share **zero tokens** in common. BM25 scores them at zero for each other, even though they refer to identical clinical concepts. This is not a ranking problem — it is a **representation problem**.

### The Research Question

Can we build a retrieval and ranking system that:

1. **Understands semantic relevance** — maps synonymous clinical concepts to similar representations
2. **Leverages behavioral signals** — uses physician interaction patterns to improve recommendations
3. **Handles multiple modalities** — combines text, image features, and structured data
4. **Serves at scale** — under 100ms p99 latency at 100K documents

### Success Metric

**NDCG@10** (Normalized Discounted Cumulative Gain at position 10) — the standard metric for ranked retrieval that weights top positions more heavily than lower ones.

### Approach

We climb the complexity ladder one rung at a time, only adding complexity when a simpler approach has demonstrably failed:

1. **BM25 baseline** → NDCG@10 = 0.61 (vocabulary mismatch is the bottleneck)
2. **Logistic regression on TF-IDF features** → 0.71 (learns feature weights, still limited by representation)
3. **Linear SVM** → 0.74 (better margin, same representation ceiling)
4. **LambdaRank** → 0.83 (learns to optimize NDCG directly, but TF-IDF features plateau)
5. **LambdaRank + ALS collaborative filtering** → 0.85 (behavioral signals help, but cold start limits gains)
6. **LambdaRank + multimodal embeddings** → 0.87 (vocabulary mismatch substantially resolved)

Each step is motivated by a specific failure of the previous approach. We never add complexity for its own sake.
