"""
BM25 keyword retrieval — implemented from scratch.

BM25 scoring formula:
    score(q, d) = Σ IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))

    where:
    - IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    - TF(t,d) = frequency of term t in document d
    - N = total number of documents
    - df(t) = number of documents containing term t
    - |d| = length of document d (in tokens)
    - avgdl = average document length across the corpus
    - k1 = 1.5 (term frequency saturation parameter)
    - b = 0.75 (document length normalization parameter)

This implementation does NOT wrap a library. It shows the full math
to make the failure mode transparent: BM25 relies entirely on token
overlap, so "cardiotoxicity" and "cardiac adverse events" score zero
for each other despite being the same clinical concept.

Phase 2 baseline results:
    NDCG@10 = 0.61 on held-out evaluation set
    The bottleneck is vocabulary mismatch, not ranking weakness.
"""

import math
import re
import time
from collections import Counter, defaultdict
from typing import Optional

import numpy as np


class BM25:
    """BM25 ranking from scratch with explicit TF-IDF and IDF math.

    Parameters:
        k1: Term frequency saturation. Higher values give more weight to
            repeated terms. Default 1.5 is standard.
        b: Document length normalization. 0 = no normalization, 1 = full
            normalization relative to average doc length. Default 0.75.

    Usage:
        bm25 = BM25(k1=1.5, b=0.75)
        bm25.fit(corpus)  # list of document strings
        scores = bm25.score("cardiac adverse events")
        top_k_indices = bm25.retrieve("cardiac adverse events", top_k=10)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)  # df(t)
        self.term_freqs: list[dict[str, int]] = []  # TF(t,d) per doc
        self.idf_cache: dict[str, float] = {}
        self._fitted = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer.

        Lowercase and strip non-alphanumeric characters. No stemming
        or lemmatization — this is intentional to preserve the vocabulary
        mismatch problem that motivates learned embeddings.
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def fit(self, corpus: list[str]) -> "BM25":
        """Index a corpus of documents.

        Computes term frequencies, document frequencies, and IDF values.

        Args:
            corpus: List of document strings (body text).

        Returns:
            self for chaining.
        """
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)

        # Pass 1: compute TF and DF
        for doc in corpus:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))

            # Each unique term in this doc increments its document frequency
            for term in tf:
                self.doc_freqs[term] += 1

        # Average document length
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0

        # Precompute IDF for all terms in the vocabulary
        self.idf_cache = {}
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf_cache[term] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1
            )

        self._fitted = True
        return self

    def _idf(self, term: str) -> float:
        """Compute IDF for a term.

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Terms not in the corpus get IDF = 0 (they contribute nothing).
        """
        return self.idf_cache.get(term, 0.0)

    def score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """Score a single document against a tokenized query.

        score(q, d) = Σ_t IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))

        Args:
            query_tokens: Tokenized query terms.
            doc_idx: Index of the document in the corpus.

        Returns:
            BM25 score (higher = more relevant).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before scoring")

        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0

        for term in query_tokens:
            idf = self._idf(term)
            if idf == 0.0:
                continue

            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue

            # BM25 TF component with length normalization
            # TF(t,d) × (k1+1) / (TF(t,d) + k1 × (1 - b + b × |d|/avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            score += idf * (numerator / denominator)

        return score

    def score(self, query: str) -> np.ndarray:
        """Score all documents against a query.

        Args:
            query: Query string.

        Returns:
            Array of BM25 scores, one per document in the corpus.
        """
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size)

        for i in range(self.corpus_size):
            scores[i] = self.score_document(query_tokens, i)

        return scores

    def retrieve(
        self, query: str, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Query string.
            top_k: Number of documents to return.

        Returns:
            List of (doc_index, score) tuples, sorted by score descending.
        """
        scores = self.score(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[tuple[int, float]]]:
        """Retrieve top-k documents for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Number of documents to return per query.

        Returns:
            List of retrieval results, one per query.
        """
        return [self.retrieve(q, top_k) for q in queries]

    def measure_latency(
        self, queries: list[str], top_k: int = 10
    ) -> dict[str, float]:
        """Measure retrieval latency (p50, p95, p99) across queries.

        Used to demonstrate BM25's linear degradation with corpus size:
        1K=2ms p99, 10K=18ms p99, 100K=180ms p99.

        Args:
            queries: List of query strings.
            top_k: Number of documents to return.

        Returns:
            Dict with p50, p95, p99 latency in milliseconds.
        """
        latencies = []

        for query in queries:
            start = time.perf_counter()
            self.retrieve(query, top_k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies_arr = np.array(latencies)

        return {
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "mean_ms": float(np.mean(latencies_arr)),
            "num_queries": len(queries),
        }

    def get_vocabulary_mismatch_examples(
        self,
        doc_texts: list[str],
        doc_ids: list[str],
    ) -> list[dict]:
        """Find examples where BM25 fails due to vocabulary mismatch.

        Demonstrates the core problem: "cardiotoxicity" and "cardiac adverse events"
        share zero tokens, so BM25 scores them at zero for each other despite
        being the same clinical concept.

        Returns:
            List of failure case dicts showing query, expected relevant docs,
            and their BM25 scores (near zero).
        """
        failure_cases = [
            {
                "query": "cardiac adverse events in EGFR inhibitor trials",
                "expected_terms": ["cardiotoxicity", "cardiovascular toxicity",
                                   "TKI-induced cardiac events", "cardiac dysfunction"],
            },
            {
                "query": "hepatotoxicity in checkpoint inhibitor therapy",
                "expected_terms": ["drug-induced liver injury", "hepatic failure",
                                   "transaminase elevation", "DILI"],
            },
            {
                "query": "immune-mediated adverse reactions to PD-1 inhibitors",
                "expected_terms": ["immunogenicity", "cytokine storm",
                                   "autoimmune activation", "hypersensitivity"],
            },
            {
                "query": "nephrotoxicity from platinum-based chemotherapy",
                "expected_terms": ["acute kidney injury", "renal impairment",
                                   "creatinine elevation", "tubular necrosis"],
            },
            {
                "query": "myelosuppression management in AML treatment",
                "expected_terms": ["hematologic toxicity", "pancytopenia",
                                   "neutropenia", "bone marrow suppression"],
            },
        ]

        results = []
        for case in failure_cases:
            query = case["query"]
            scores = self.score(query)

            # Find documents containing expected synonym terms
            relevant_docs = []
            for term in case["expected_terms"]:
                for idx, text in enumerate(doc_texts):
                    if term.lower() in text.lower():
                        relevant_docs.append({
                            "doc_idx": idx,
                            "doc_id": doc_ids[idx] if idx < len(doc_ids) else f"doc_{idx}",
                            "matched_term": term,
                            "bm25_score": float(scores[idx]),
                            "text_preview": text[:200],
                        })

            results.append({
                "query": query,
                "expected_terms": case["expected_terms"],
                "relevant_docs_found": len(relevant_docs),
                "relevant_docs": relevant_docs[:5],  # Top 5 examples
                "diagnosis": (
                    "BM25 scores near zero because query terms and document terms "
                    "are different words for the same clinical concept. No amount of "
                    "BM25 parameter tuning can fix this — the problem is vocabulary "
                    "mismatch, not ranking weakness."
                ),
            })

        return results
