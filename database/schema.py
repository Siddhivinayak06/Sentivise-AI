"""Database Schema Definition and Table Initialization for Sentiment System."""
import sqlite3
from database.connection import get_db

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL,
    prob_pos REAL,
    prob_neg REAL,
    model_used TEXT NOT NULL,
    rules_triggered TEXT,
    recommendation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS batch_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    total_rows INTEGER NOT NULL,
    valid_rows INTEGER NOT NULL,
    invalid_rows INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS batch_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    import_id INTEGER NOT NULL,
    original_index INTEGER,
    review_text TEXT NOT NULL,
    product_id TEXT,
    product_name TEXT,
    review_date TEXT,
    rating REAL,
    sentiment TEXT NOT NULL,
    confidence REAL,
    model_used TEXT,
    rules_triggered TEXT,
    recommendation TEXT,
    analysis_status TEXT DEFAULT 'Success',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(import_id) REFERENCES batch_imports(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'Medium',
    status TEXT NOT NULL DEFAULT 'Open',
    assigned_to TEXT DEFAULT 'Unassigned',
    priority_score REAL DEFAULT 0.0,
    rationale TEXT,
    recommended_action TEXT,
    rule_identifiers TEXT,
    affected_products_count INTEGER DEFAULT 1,
    supporting_reviews_count INTEGER DEFAULT 1,
    first_seen TEXT,
    last_seen TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS issue_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL,
    author TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(issue_id) REFERENCES issues(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    description TEXT NOT NULL,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_batch_import_id ON batch_reviews(import_id);
CREATE INDEX IF NOT EXISTS idx_batch_sentiment ON batch_reviews(sentiment);
CREATE INDEX IF NOT EXISTS idx_batch_product ON batch_reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);
"""

def initialize_database():
    """Initializes tables and indexes if they do not exist."""
    with get_db() as conn:
        conn.executescript(SCHEMA_SQL)
