"""
Logistic regression baseline for document relevance classification.

Feature vector:
    - TF-IDF cosine similarity between query and document
    - BM25 score
    - Query term overlap ratio
    - Document length (normalized)
    - Document type one-hot encoding (5 types)
    - Physician specialty match flag (1 if physician and doc share specialty)

Grid search over L2 regularization parameter C.

Phase 3 results:
    NDCG@10 = 0.71 — better than BM25 (0.61) because it learns feature weights
    from labeled data, but still limited by TF-IDF feature representation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.retrieval.bm25 import BM25


class LogisticBaselineRanker:
    """Logistic regression ranker on handcrafted features.

    Trains a binary classifier (relevant vs. not relevant) and uses
    predicted probability as the ranking score.

    The feature ceiling becomes apparent here: adding more TF-IDF features
    does not improve NDCG beyond ~0.75 regardless of model sophistication.
    The bottleneck is feature representation, not model complexity.
    """

    def __init__(self, bm25: BM25):
        self.bm25 = bm25
        self.model: LogisticRegression | None = None
        self.scaler = StandardScaler()
        self.doc_type_encoder = OneHotEncoder(
            categories=[["trial_report", "case_study", "drug_interaction",
                         "pathology_report", "cohort_summary"]],
            sparse_output=False,
            handle_unknown="ignore",
        )
        self._fitted = False

    def extract_features(
        self,
        query_texts: list[str],
        doc_texts: list[str],
        doc_types: list[str],
        query_specialties: list[str],
        doc_specialties: list[str],
    ) -> np.ndarray:
        """Extract feature vectors for query-document pairs.

        Features (per pair):
            0: TF-IDF cosine similarity
            1: BM25 score
            2: Query term overlap ratio
            3: Document length (word count, normalized)
            4-8: Document type one-hot (5 dims)
            9: Specialty match flag

        Args:
            query_texts: Query strings.
            doc_texts: Document body texts (same length as query_texts).
            doc_types: Document types (same length).
            query_specialties: Physician specialties for each query.
            doc_specialties: Document specialties (same length).

        Returns:
            Feature matrix of shape (n_pairs, 10).
        """
        n = len(query_texts)
        features = np.zeros((n, 10), dtype=np.float32)

        for i in range(n):
            q_tokens = set(self.bm25.tokenize(query_texts[i]))
            d_tokens = set(self.bm25.tokenize(doc_texts[i]))

            # Feature 0: TF-IDF cosine similarity (approximate via token overlap)
            if len(q_tokens) > 0 and len(d_tokens) > 0:
                intersection = q_tokens & d_tokens
                features[i, 0] = len(intersection) / (
                    np.sqrt(len(q_tokens)) * np.sqrt(len(d_tokens))
                )

            # Feature 1: BM25 score (requires the doc to be in the fitted corpus)
            # We compute a quick BM25-like score inline
            doc_token_list = self.bm25.tokenize(doc_texts[i])
            doc_len = len(doc_token_list)
            if doc_len > 0 and self.bm25._fitted:
                score = 0.0
                from collections import Counter
                tf_counter = Counter(doc_token_list)
                for term in q_tokens:
                    idf = self.bm25._idf(term)
                    tf = tf_counter.get(term, 0)
                    if tf > 0 and idf > 0:
                        num = tf * (self.bm25.k1 + 1)
                        denom = tf + self.bm25.k1 * (
                            1 - self.bm25.b + self.bm25.b * doc_len / max(self.bm25.avgdl, 1)
                        )
                        score += idf * (num / denom)
                features[i, 1] = score

            # Feature 2: Query term overlap ratio
            if len(q_tokens) > 0:
                features[i, 2] = len(q_tokens & d_tokens) / len(q_tokens)

            # Feature 3: Document length (word count)
            features[i, 3] = len(doc_texts[i].split())

        # Features 4-8: Document type one-hot
        doc_type_array = np.array(doc_types).reshape(-1, 1)
        doc_type_encoded = self.doc_type_encoder.fit_transform(doc_type_array)
        features[:, 4:9] = doc_type_encoded

        # Feature 9: Specialty match flag
        for i in range(n):
            features[i, 9] = float(query_specialties[i] == doc_specialties[i])

        return features

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        cv_folds: int = 5,
    ) -> dict:
        """Train logistic regression with grid search over regularization.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            labels: Binary relevance labels (0 or 1).
            cv_folds: Number of cross-validation folds.

        Returns:
            Dict with best parameters and CV scores.
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Grid search over C parameter
        param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

        grid_search = GridSearchCV(
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            ),
            param_grid,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            return_train_score=True,
        )

        grid_search.fit(features_scaled, labels)
        self.model = grid_search.best_estimator_
        self._fitted = True

        return {
            "best_C": grid_search.best_params_["C"],
            "best_cv_auc": grid_search.best_score_,
            "cv_results": {
                "mean_train_score": grid_search.cv_results_["mean_train_score"].tolist(),
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "params": grid_search.cv_results_["params"],
            },
        }

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores (probabilities) for ranking.

        Args:
            features: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted probabilities (used as ranking scores).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict_scores()")

        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]

    def rank_documents(
        self,
        query_text: str,
        doc_texts: list[str],
        doc_types: list[str],
        query_specialty: str,
        doc_specialties: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rank documents for a single query.

        Returns:
            List of (doc_index, score) tuples, sorted by score descending.
        """
        n = len(doc_texts)
        features = self.extract_features(
            [query_text] * n,
            doc_texts,
            doc_types,
            [query_specialty] * n,
            doc_specialties,
        )
        scores = self.predict_scores(features)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
