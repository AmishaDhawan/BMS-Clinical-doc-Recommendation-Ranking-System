"""
Data loader for BMS Clinical Document Recommendation System.

Loads data from SQLite using the SQL queries defined in sql_queries.py.
Returns pandas DataFrames and scipy sparse matrices for downstream use.

This module is the single entry point for all data access — no other module
should query SQLite directly.
"""

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from src.data_pipeline.sql_queries import (
    COLD_START_DOCUMENTS_QUERY,
    DOCUMENTS_QUERY,
    INTERACTION_MATRIX_QUERY,
    NDCG_EVALUATION_QUERY,
    PHYSICIANS_QUERY,
    RELEVANCE_JUDGMENTS_QUERY,
    SEARCH_QUERIES_WITH_CONTEXT,
    WEEKLY_CTR_QUERY,
)


class ClinicalDataLoader:
    """Loads clinical document data from SQLite for model training and evaluation.

    All SQL queries are defined in sql_queries.py and called here — no dead code.

    Usage:
        loader = ClinicalDataLoader("data/synthetic/clinical_docs_1k.db")
        docs_df = loader.load_documents()
        interaction_matrix = loader.build_interaction_matrix()
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with math functions enabled."""
        conn = sqlite3.connect(self.db_path)
        # Register LOG and POWER functions for NDCG computation
        conn.create_function("LOG", 1, lambda x: np.log(x) if x > 0 else 0.0)
        conn.create_function("POWER", 2, lambda x, y: x ** y)
        return conn

    def load_documents(self) -> pd.DataFrame:
        """Load all documents with metadata.

        Returns:
            DataFrame with columns: doc_id, title, specialty, doc_type,
            body_text, created_at, icd_codes
        """
        conn = self._get_connection()
        df = pd.read_sql_query(DOCUMENTS_QUERY, conn)
        conn.close()
        return df

    def load_physicians(self) -> pd.DataFrame:
        """Load all physician profiles.

        Returns:
            DataFrame with columns: physician_id, specialty, years_experience
        """
        conn = self._get_connection()
        df = pd.read_sql_query(PHYSICIANS_QUERY, conn)
        conn.close()
        return df

    def load_search_queries(self) -> pd.DataFrame:
        """Load search queries with physician context.

        Returns:
            DataFrame with columns: query_id, physician_id, query_text,
            timestamp, physician_specialty, years_experience
        """
        conn = self._get_connection()
        df = pd.read_sql_query(SEARCH_QUERIES_WITH_CONTEXT, conn)
        conn.close()
        return df

    def load_relevance_judgments(self) -> pd.DataFrame:
        """Load relevance judgments for evaluation.

        Returns:
            DataFrame with columns: query_id, doc_id, relevance_score,
            query_text, doc_title, doc_specialty, doc_type
        """
        conn = self._get_connection()
        df = pd.read_sql_query(RELEVANCE_JUDGMENTS_QUERY, conn)
        conn.close()
        return df

    def load_interaction_matrix_df(self) -> pd.DataFrame:
        """Load interaction matrix with confidence weights as a DataFrame.

        Uses the CTE-based query with confidence weighting:
        c_ui = 1 + 40 * weighted_score
        where weighted_score = saves×3 + clicks×2 + views×1

        Returns:
            DataFrame with columns: physician_id, doc_id, weighted_score,
            confidence, total_interactions, max_dwell_time, avg_dwell_time,
            rank_per_physician
        """
        conn = self._get_connection()
        df = pd.read_sql_query(INTERACTION_MATRIX_QUERY, conn)
        conn.close()
        return df

    def build_interaction_matrix(self) -> tuple[sparse.csr_matrix, list[str], list[str]]:
        """Build sparse user-document interaction matrix for ALS.

        Returns:
            Tuple of (sparse_matrix, physician_ids, doc_ids) where:
            - sparse_matrix: scipy CSR matrix of shape (n_physicians, n_docs)
              with confidence-weighted interaction scores
            - physician_ids: list of physician IDs (row index)
            - doc_ids: list of document IDs (column index)
        """
        interaction_df = self.load_interaction_matrix_df()

        if interaction_df.empty:
            return sparse.csr_matrix((0, 0)), [], []

        # Create ID-to-index mappings
        physician_ids = sorted(interaction_df["physician_id"].unique().tolist())
        doc_ids = sorted(interaction_df["doc_id"].unique().tolist())

        physician_idx = {pid: i for i, pid in enumerate(physician_ids)}
        doc_idx = {did: i for i, did in enumerate(doc_ids)}

        # Build sparse matrix with confidence weights
        rows = interaction_df["physician_id"].map(physician_idx).values
        cols = interaction_df["doc_id"].map(doc_idx).values
        values = interaction_df["confidence"].values.astype(np.float32)

        matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(physician_ids), len(doc_ids)),
        )

        return matrix, physician_ids, doc_ids

    def load_cold_start_documents(self) -> pd.DataFrame:
        """Identify cold start documents (fewer than 5 interactions).

        Uses the CTE-based cold start detection query.

        Returns:
            DataFrame with columns: doc_id, title, specialty, doc_type,
            interaction_count, unique_physicians, is_cold_start
        """
        conn = self._get_connection()
        df = pd.read_sql_query(COLD_START_DOCUMENTS_QUERY, conn)
        conn.close()
        return df

    def load_weekly_ctr(self) -> pd.DataFrame:
        """Load weekly click-through rate data for novelty monitoring.

        Returns:
            DataFrame with columns: year, week_num, click_count, view_count,
            ctr, active_physicians, docs_interacted
        """
        conn = self._get_connection()
        df = pd.read_sql_query(WEEKLY_CTR_QUERY, conn)
        conn.close()
        return df

    def load_ndcg_evaluation_data(self) -> pd.DataFrame:
        """Load NDCG evaluation data with discount weights.

        Uses the window function query to compute ideal DCG for normalization.

        Returns:
            DataFrame with columns: query_id, query_text, doc_id,
            relevance_score, ideal_rank, discount, gain, dcg_contribution,
            cumulative_dcg
        """
        conn = self._get_connection()
        df = pd.read_sql_query(NDCG_EVALUATION_QUERY, conn)
        conn.close()
        return df

    def compute_sparsity(self) -> dict:
        """Compute interaction matrix sparsity statistics.

        Returns:
            Dict with sparsity metrics at the current scale.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        n_docs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM physicians")
        n_physicians = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(DISTINCT physician_id || '|' || doc_id) FROM interactions"
        )
        n_unique_pairs = cursor.fetchone()[0]

        conn.close()

        total_possible = n_docs * n_physicians
        sparsity = 1 - (n_unique_pairs / total_possible) if total_possible > 0 else 1.0

        return {
            "n_documents": n_docs,
            "n_physicians": n_physicians,
            "total_possible_pairs": total_possible,
            "observed_pairs": n_unique_pairs,
            "sparsity": sparsity,
            "sparsity_pct": f"{sparsity * 100:.2f}%",
        }

    def get_dataset_summary(self) -> dict:
        """Get a summary of the dataset for exploration notebooks."""
        conn = self._get_connection()
        cursor = conn.cursor()

        summary = {}
        for table in ["documents", "physicians", "interactions",
                       "search_queries", "query_document_relevance"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            summary[f"{table}_count"] = cursor.fetchone()[0]

        # Document type distribution
        cursor.execute(
            "SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type ORDER BY cnt DESC"
        )
        summary["doc_type_distribution"] = dict(cursor.fetchall())

        # Specialty distribution
        cursor.execute(
            "SELECT specialty, COUNT(*) as cnt FROM documents GROUP BY specialty ORDER BY cnt DESC"
        )
        summary["specialty_distribution"] = dict(cursor.fetchall())

        # Interaction type distribution
        cursor.execute(
            "SELECT interaction_type, COUNT(*) as cnt FROM interactions GROUP BY interaction_type ORDER BY cnt DESC"
        )
        summary["interaction_type_distribution"] = dict(cursor.fetchall())

        # Relevance score distribution
        cursor.execute(
            "SELECT relevance_score, COUNT(*) as cnt FROM query_document_relevance GROUP BY relevance_score ORDER BY relevance_score"
        )
        summary["relevance_distribution"] = dict(cursor.fetchall())

        conn.close()
        return summary
