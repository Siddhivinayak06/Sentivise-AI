"""Database Repository Layer providing safe parameterized access to SQLite."""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from database.connection import get_db

class Repository:
    @staticmethod
    def save_analysis(
        review_text: str,
        sentiment: str,
        confidence: float,
        prob_pos: Optional[float],
        prob_neg: Optional[float],
        model_used: str,
        rules_triggered: List[str],
        recommendation: str
    ) -> int:
        """Saves a single review analysis to history."""
        rules_json = json.dumps(rules_triggered) if isinstance(rules_triggered, list) else str(rules_triggered)
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO analysis_history (
                    review_text, sentiment, confidence, prob_pos, prob_neg,
                    model_used, rules_triggered, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (review_text, sentiment, confidence, prob_pos, prob_neg, model_used, rules_json, recommendation)
            )
            analysis_id = cursor.lastrowid
            Repository.log_audit_event(
                action_type="SINGLE_ANALYSIS",
                entity_type="ANALYSIS",
                entity_id=str(analysis_id),
                description=f"Analyzed single review using {model_used} -> {sentiment} ({confidence:.2f})"
            )
            return analysis_id

    @staticmethod
    def get_recent_analyses(limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent single analyses."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def create_batch_import(filename: str, total_rows: int, valid_rows: int, invalid_rows: int) -> int:
        """Records a new batch import event."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO batch_imports (filename, total_rows, valid_rows, invalid_rows)
                VALUES (?, ?, ?, ?)
                """,
                (filename, total_rows, valid_rows, invalid_rows)
            )
            import_id = cursor.lastrowid
            Repository.log_audit_event(
                action_type="BATCH_IMPORT",
                entity_type="BATCH",
                entity_id=str(import_id),
                description=f"Imported batch file {filename}: {valid_rows} valid rows out of {total_rows}"
            )
            return import_id

    @staticmethod
    def insert_batch_reviews(reviews_data: List[Dict[str, Any]]) -> None:
        """Batch inserts analyzed review rows."""
        if not reviews_data:
            return
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO batch_reviews (
                    import_id, original_index, review_text, product_id, product_name,
                    review_date, rating, sentiment, confidence, model_used,
                    rules_triggered, recommendation, analysis_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.get("import_id"),
                        r.get("original_index"),
                        r.get("review_text"),
                        r.get("product_id"),
                        r.get("product_name"),
                        r.get("review_date"),
                        r.get("rating"),
                        r.get("sentiment"),
                        r.get("confidence"),
                        r.get("model_used"),
                        json.dumps(r.get("rules_triggered", [])) if isinstance(r.get("rules_triggered"), list) else str(r.get("rules_triggered", "")),
                        r.get("recommendation"),
                        r.get("analysis_status", "Success")
                    )
                    for r in reviews_data
                ]
            )

    @staticmethod
    def get_batch_imports() -> List[Dict[str, Any]]:
        """Retrieves list of all batch imports."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM batch_imports ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_batch_reviews(
        import_id: Optional[int] = None,
        sentiment: Optional[str] = None,
        product_id: Optional[str] = None,
        limit: int = 2000
    ) -> List[Dict[str, Any]]:
        """Queries batch reviews with flexible filters."""
        query = "SELECT * FROM batch_reviews WHERE 1=1"
        params: List[Any] = []
        if import_id is not None:
            query += " AND import_id = ?"
            params.append(import_id)
        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def create_issue(
        title: str,
        category: str,
        severity: str = "Medium",
        status: str = "Open",
        assigned_to: str = "Unassigned",
        priority_score: float = 0.0,
        rationale: str = "",
        recommended_action: str = "",
        rule_identifiers: str = "",
        affected_products_count: int = 1,
        supporting_reviews_count: int = 1,
        first_seen: Optional[str] = None,
        last_seen: Optional[str] = None
    ) -> int:
        """Creates a new tracked issue in the Decision Center."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO issues (
                    title, category, severity, status, assigned_to, priority_score,
                    rationale, recommended_action, rule_identifiers, affected_products_count,
                    supporting_reviews_count, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title, category, severity, status, assigned_to, priority_score,
                    rationale, recommended_action, rule_identifiers, affected_products_count,
                    supporting_reviews_count, first_seen, last_seen
                )
            )
            issue_id = cursor.lastrowid
            Repository.log_audit_event(
                action_type="ISSUE_CREATED",
                entity_type="ISSUE",
                entity_id=str(issue_id),
                description=f"Created issue '{title}' ({category}, Priority: {priority_score:.1f})"
            )
            return issue_id

    @staticmethod
    def get_issues(status: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves issues with optional status/category filter."""
        query = "SELECT * FROM issues WHERE 1=1"
        params: List[Any] = []
        if status and status != "All":
            query += " AND status = ?"
            params.append(status)
        if category and category != "All":
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY priority_score DESC, id DESC"
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_issue_by_id(issue_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a specific issue by ID."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM issues WHERE id = ?", (issue_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def update_issue_status(issue_id: int, new_status: str, assigned_to: Optional[str] = None, author: str = "System", note: Optional[str] = None) -> bool:
        """Updates issue status, assignee, and appends a transition note."""
        now = datetime.now().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            if assigned_to:
                cursor.execute(
                    "UPDATE issues SET status = ?, assigned_to = ?, updated_at = ? WHERE id = ?",
                    (new_status, assigned_to, now, issue_id)
                )
            else:
                cursor.execute(
                    "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, now, issue_id)
                )
            if cursor.rowcount > 0:
                note_text = f"Status changed to {new_status}"
                if assigned_to:
                    note_text += f", assigned to {assigned_to}"
                if note:
                    note_text += f": {note}"
                cursor.execute(
                    "INSERT INTO issue_notes (issue_id, author, note) VALUES (?, ?, ?)",
                    (issue_id, author, note_text)
                )
                Repository.log_audit_event(
                    action_type="STATUS_CHANGE",
                    entity_type="ISSUE",
                    entity_id=str(issue_id),
                    description=f"Issue #{issue_id} status updated to {new_status} by {author}"
                )
                return True
            return False

    @staticmethod
    def add_issue_note(issue_id: int, author: str, note: str) -> int:
        """Adds a note to an issue."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO issue_notes (issue_id, author, note) VALUES (?, ?, ?)",
                (issue_id, author, note)
            )
            note_id = cursor.lastrowid
            cursor.execute("UPDATE issues SET updated_at = ? WHERE id = ?", (datetime.now().isoformat(), issue_id))
            return note_id

    @staticmethod
    def get_issue_notes(issue_id: int) -> List[Dict[str, Any]]:
        """Retrieves all notes for a specific issue."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM issue_notes WHERE issue_id = ? ORDER BY id ASC", (issue_id,))
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def log_audit_event(
        action_type: str,
        entity_type: str,
        entity_id: Optional[str],
        description: str,
        metadata_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """Appends an event to the immutable audit trail."""
        meta_str = json.dumps(metadata_dict) if metadata_dict else None
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_logs (action_type, entity_type, entity_id, description, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (action_type, entity_type, entity_id, description, meta_str)
            )
            return cursor.lastrowid

    @staticmethod
    def get_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves audit trail entries."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def get_system_kpis() -> Dict[str, Any]:
        """Calculates global KPIs from persistent records."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            single_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM batch_reviews")
            batch_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM batch_reviews WHERE sentiment = 'Positive'")
            batch_pos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM batch_reviews WHERE sentiment = 'Negative'")
            batch_neg = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE sentiment = 'Positive'")
            single_pos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE sentiment = 'Negative'")
            single_neg = cursor.fetchone()[0]

            total_analyzed = single_count + batch_count
            total_pos = single_pos + batch_pos
            total_neg = single_neg + batch_neg

            cursor.execute("SELECT AVG(rating) FROM batch_reviews WHERE rating IS NOT NULL")
            avg_rating = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT product_id) FROM batch_reviews WHERE product_id IS NOT NULL AND product_id != ''")
            distinct_products = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM issues WHERE status = 'Open' OR status = 'Investigating'")
            active_issues = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM issues WHERE severity = 'Critical' OR severity = 'High'")
            high_sev_issues = cursor.fetchone()[0]

            return {
                "total_analyzed": total_analyzed,
                "total_positive": total_pos,
                "total_negative": total_neg,
                "positive_percentage": (total_pos / total_analyzed * 100) if total_analyzed > 0 else 0.0,
                "negative_percentage": (total_neg / total_analyzed * 100) if total_analyzed > 0 else 0.0,
                "avg_rating": round(avg_rating, 2) if avg_rating is not None else None,
                "distinct_products": distinct_products,
                "active_issues": active_issues,
                "high_priority_issues": high_sev_issues
            }

    @staticmethod
    def clear_table(table_name: str) -> bool:
        """Safely clears a specific table and records audit log."""
        allowed = ["analysis_history", "batch_reviews", "batch_imports", "issues", "issue_notes", "audit_logs"]
        if table_name not in allowed:
            raise ValueError(f"Invalid table: {table_name}")
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            Repository.log_audit_event(
                action_type="DATA_DELETION",
                entity_type="TABLE",
                entity_id=table_name,
                description=f"Purged all rows from table {table_name}"
            )
            return True
