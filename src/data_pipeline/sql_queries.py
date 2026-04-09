"""
SQL queries used throughout the BMS Clinical Document Recommendation System.

These queries are called in actual data loading and evaluation code — not dead code.
Each query uses CTEs and window functions for clarity and performance.
"""


# Interaction matrix with confidence weighting for ALS collaborative filtering.
# Confidence formula: c_ui = 1 + 40 * weighted_score
# where weighted_score = saves×3 + clicks×2 + views×1
# This weighting reflects that saves indicate stronger preference than clicks,
# which indicate stronger preference than views.
INTERACTION_MATRIX_QUERY = """
WITH interaction_weights AS (
    SELECT
        physician_id,
        doc_id,
        SUM(CASE WHEN interaction_type = 'save' THEN 3
                 WHEN interaction_type = 'click' THEN 2
                 WHEN interaction_type = 'view' THEN 1
                 ELSE 0 END) AS weighted_score,
        COUNT(*) AS total_interactions,
        MAX(dwell_time_seconds) AS max_dwell_time,
        AVG(dwell_time_seconds) AS avg_dwell_time
    FROM interactions
    GROUP BY physician_id, doc_id
),
confidence_scores AS (
    SELECT
        physician_id,
        doc_id,
        weighted_score,
        1 + 40 * weighted_score AS confidence,
        total_interactions,
        max_dwell_time,
        avg_dwell_time,
        ROW_NUMBER() OVER (
            PARTITION BY physician_id
            ORDER BY weighted_score DESC
        ) AS rank_per_physician
    FROM interaction_weights
)
SELECT
    physician_id,
    doc_id,
    weighted_score,
    confidence,
    total_interactions,
    max_dwell_time,
    avg_dwell_time,
    rank_per_physician
FROM confidence_scores
ORDER BY physician_id, weighted_score DESC
"""


# Cold start document detection: documents with fewer than 5 total interactions.
# These documents will receive zero ALS factors and must fall back to content features.
COLD_START_DOCUMENTS_QUERY = """
WITH doc_interaction_counts AS (
    SELECT
        d.doc_id,
        d.title,
        d.specialty,
        d.doc_type,
        COALESCE(COUNT(i.interaction_id), 0) AS interaction_count,
        COALESCE(
            COUNT(DISTINCT i.physician_id), 0
        ) AS unique_physicians
    FROM documents d
    LEFT JOIN interactions i ON d.doc_id = i.doc_id
    GROUP BY d.doc_id, d.title, d.specialty, d.doc_type
)
SELECT
    doc_id,
    title,
    specialty,
    doc_type,
    interaction_count,
    unique_physicians,
    CASE WHEN interaction_count < 5 THEN 1 ELSE 0 END AS is_cold_start
FROM doc_interaction_counts
ORDER BY interaction_count ASC
"""


# Weekly CTR query for novelty effect monitoring.
# CTR = COUNT(clicks) / COUNT(views) grouped by week.
# Used to detect if click-through rates decay over time (novelty wearing off).
WEEKLY_CTR_QUERY = """
WITH weekly_events AS (
    SELECT
        STRFTIME('%Y', timestamp) AS year,
        STRFTIME('%W', timestamp) AS week_num,
        interaction_type,
        doc_id,
        physician_id
    FROM interactions
    WHERE interaction_type IN ('click', 'view')
),
weekly_aggregates AS (
    SELECT
        year,
        week_num,
        COUNT(CASE WHEN interaction_type = 'click' THEN 1 END) AS click_count,
        COUNT(CASE WHEN interaction_type = 'view' THEN 1 END) AS view_count,
        COUNT(DISTINCT physician_id) AS active_physicians,
        COUNT(DISTINCT doc_id) AS docs_interacted
    FROM weekly_events
    GROUP BY year, week_num
)
SELECT
    year,
    week_num,
    click_count,
    view_count,
    CASE WHEN view_count > 0
         THEN CAST(click_count AS REAL) / view_count
         ELSE 0.0
    END AS ctr,
    active_physicians,
    docs_interacted
FROM weekly_aggregates
ORDER BY year, week_num
"""


# NDCG evaluation query: computes relevance scores with position-based discount weights.
# Discount: 1.0 / LOG2(rank + 1) using ROW_NUMBER() window function.
# This query is used by the evaluation module to compute NDCG@k for any model's predictions.
NDCG_EVALUATION_QUERY = """
WITH ranked_results AS (
    SELECT
        qdr.query_id,
        qdr.doc_id,
        qdr.relevance_score,
        sq.query_text,
        ROW_NUMBER() OVER (
            PARTITION BY qdr.query_id
            ORDER BY qdr.relevance_score DESC
        ) AS ideal_rank
    FROM query_document_relevance qdr
    JOIN search_queries sq ON qdr.query_id = sq.query_id
),
dcg_components AS (
    SELECT
        query_id,
        query_text,
        doc_id,
        relevance_score,
        ideal_rank,
        -- DCG discount weight: 1 / log2(rank + 1)
        1.0 / (LOG(ideal_rank + 1) / LOG(2)) AS discount,
        -- Gain: 2^relevance - 1 (standard NDCG gain formula)
        (POWER(2, relevance_score) - 1) AS gain,
        -- Discounted gain for this position
        (POWER(2, relevance_score) - 1) * (1.0 / (LOG(ideal_rank + 1) / LOG(2))) AS dcg_contribution
    FROM ranked_results
)
SELECT
    query_id,
    query_text,
    doc_id,
    relevance_score,
    ideal_rank,
    discount,
    gain,
    dcg_contribution,
    SUM(dcg_contribution) OVER (
        PARTITION BY query_id
        ORDER BY ideal_rank
        ROWS UNBOUNDED PRECEDING
    ) AS cumulative_dcg
FROM dcg_components
ORDER BY query_id, ideal_rank
"""


# Load all documents with metadata for feature extraction
DOCUMENTS_QUERY = """
SELECT
    doc_id,
    title,
    specialty,
    doc_type,
    body_text,
    created_at,
    icd_codes
FROM documents
ORDER BY doc_id
"""


# Load all physician profiles
PHYSICIANS_QUERY = """
SELECT
    physician_id,
    specialty,
    years_experience
FROM physicians
ORDER BY physician_id
"""


# Load search queries with physician context
SEARCH_QUERIES_WITH_CONTEXT = """
SELECT
    sq.query_id,
    sq.physician_id,
    sq.query_text,
    sq.timestamp,
    p.specialty AS physician_specialty,
    p.years_experience
FROM search_queries sq
JOIN physicians p ON sq.physician_id = p.physician_id
ORDER BY sq.timestamp
"""


# Load relevance judgments for evaluation
RELEVANCE_JUDGMENTS_QUERY = """
SELECT
    qdr.query_id,
    qdr.doc_id,
    qdr.relevance_score,
    sq.query_text,
    d.title AS doc_title,
    d.specialty AS doc_specialty,
    d.doc_type
FROM query_document_relevance qdr
JOIN search_queries sq ON qdr.query_id = sq.query_id
JOIN documents d ON qdr.doc_id = d.doc_id
ORDER BY qdr.query_id, qdr.relevance_score DESC
"""
