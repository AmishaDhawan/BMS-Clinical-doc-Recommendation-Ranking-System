-- BMS Clinical Document Recommendation System
-- Database schema for physician-document interactions
-- SQLite-compatible schema

CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    specialty TEXT,
    doc_type TEXT CHECK(doc_type IN ('trial_report','case_study','drug_interaction','pathology_report','cohort_summary')),
    body_text TEXT,
    created_at TIMESTAMP,
    icd_codes TEXT
);

CREATE TABLE physicians (
    physician_id TEXT PRIMARY KEY,
    specialty TEXT,
    years_experience INTEGER
);

CREATE TABLE interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    physician_id TEXT REFERENCES physicians(physician_id),
    doc_id TEXT REFERENCES documents(doc_id),
    interaction_type TEXT CHECK(interaction_type IN ('view','click','save','dwell')),
    dwell_time_seconds REAL,
    timestamp TIMESTAMP,
    query_id TEXT
);

CREATE TABLE search_queries (
    query_id TEXT PRIMARY KEY,
    physician_id TEXT REFERENCES physicians(physician_id),
    query_text TEXT,
    timestamp TIMESTAMP
);

CREATE TABLE query_document_relevance (
    query_id TEXT REFERENCES search_queries(query_id),
    doc_id TEXT REFERENCES documents(doc_id),
    relevance_score INTEGER CHECK(relevance_score IN (0,1,2,3)),
    PRIMARY KEY (query_id, doc_id)
);

-- Indices for common access patterns
CREATE INDEX idx_interactions_physician ON interactions(physician_id);
CREATE INDEX idx_interactions_doc ON interactions(doc_id);
CREATE INDEX idx_interactions_query ON interactions(query_id);
CREATE INDEX idx_interactions_type ON interactions(interaction_type);
CREATE INDEX idx_search_queries_physician ON search_queries(physician_id);
CREATE INDEX idx_query_doc_rel_query ON query_document_relevance(query_id);
CREATE INDEX idx_query_doc_rel_doc ON query_document_relevance(doc_id);
CREATE INDEX idx_documents_specialty ON documents(specialty);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
