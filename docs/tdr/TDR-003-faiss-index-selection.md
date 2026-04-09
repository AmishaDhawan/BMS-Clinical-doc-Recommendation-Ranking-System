# TDR-003: FAISS Index Selection for Embedding Retrieval

## Status: Deferred to Part 2

## Context

With 128-dim multimodal embeddings for up to 100K documents, we need a vector similarity search index that supports:
- Sub-100ms p99 retrieval latency
- Approximate nearest neighbor search for the retrieval stage
- Efficient memory usage (3 modalities × 100K × 128 × 4 bytes = 153.6MB total)

This TDR documents the architectural decision space. Implementation and benchmarking are deferred to Part 2.

## Embedding Dimensions (established in Part 1)

- Text embeddings: 128-dim (BioBERT + projection head)
- Image embeddings: 128-dim (2048-dim ResNet features → MLP → 128)
- Structured embeddings: 128-dim (ICD codes + lab values + biomarkers → MLP → 128)
- All embeddings are L2-normalized before indexing

## Index Options Under Consideration

### Flat (exact search)
- Brute-force cosine similarity over all vectors
- Exact results, no approximation error
- At 100K × 128-dim: ~50ms per query (may exceed latency budget)
- Suitable for 1K-10K scale

### IVF (Inverted File Index)
- Clusters vectors into Voronoi cells, searches only nearby cells
- Trade-off: nprobe controls accuracy vs. speed
- At 100K with nlist=256, nprobe=16: ~5ms per query
- Risk: cluster boundaries may split similar clinical concepts

### HNSW (Hierarchical Navigable Small World)
- Graph-based ANN with logarithmic search complexity
- Higher memory overhead but excellent recall at high speed
- At 100K: ~2ms per query with 95%+ recall
- Best candidate for production serving

### IVF-PQ (Product Quantization)
- Compresses vectors via product quantization, reducing memory 4-8x
- At 100K with PQ32: memory drops from 50MB to ~12MB
- Higher approximation error — needs careful calibration
- May be necessary at 100K+ scale if memory is constrained

## Decision Criteria (to be evaluated in Part 2)
1. Latency at p99 < 100ms
2. Recall@100 > 95%
3. Memory footprint fits in SageMaker instance budget
4. Index build time acceptable for daily reindexing

## Dependencies from Part 1
- Embedding dimension: **128-dim per modality**
- L2 normalization: all embeddings are unit vectors (inner product = cosine similarity)
- Total embedding store: 153.6MB at 100K documents
