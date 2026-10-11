"""Unit tests for SQLite schema, repository persistence, and audit logging."""
import pytest
from database.schema import initialize_database
from database.repository import Repository

@pytest.fixture(autouse=True)
def setup_db():
    initialize_database()

def test_save_and_retrieve_analysis():
    text = "Great sound and build quality."
    saved_id = Repository.save_analysis(
        review_text=text,
        sentiment="Positive",
        confidence=0.92,
        prob_pos=0.92,
        prob_neg=0.08,
        model_used="Logistic Regression (Best)",
        rules_triggered=["RULE_POS_CHAMP_01"],
        recommendation="Brand Ambassador Invite"
    )
    assert saved_id > 0

    recent = Repository.get_recent_analyses(limit=5)
    assert len(recent) > 0
    found = any(r["id"] == saved_id for r in recent)
    assert found

def test_issue_lifecycle_and_notes():
    issue_id = Repository.create_issue(
        title="Test Charger Overheat Issue",
        category="Product Quality",
        severity="High",
        status="Open",
        priority_score=75.0,
        rationale="Repeated thermal complaints",
        recommended_action="Inspect thermal fuses"
    )
    assert issue_id > 0

    # Retrieve
    issue = Repository.get_issue_by_id(issue_id)
    assert issue is not None
    assert issue["status"] == "Open"

    # Transition status
    updated = Repository.update_issue_status(
        issue_id=issue_id,
        new_status="Investigating",
        assigned_to="QA Team",
        author="Auditor",
        note="Assigned to senior QA engineer"
    )
    assert updated

    # Check updated status and note
    issue_after = Repository.get_issue_by_id(issue_id)
    assert issue_after["status"] == "Investigating"
    assert issue_after["assigned_to"] == "QA Team"

    notes = Repository.get_issue_notes(issue_id)
    assert len(notes) > 0
    assert "Assigned to senior QA engineer" in notes[-1]["note"]

def test_audit_logs():
    log_id = Repository.log_audit_event(
        action_type="TEST_ACTION",
        entity_type="SYSTEM",
        entity_id="0",
        description="Test audit log entry"
    )
    assert log_id > 0
    logs = Repository.get_audit_logs(limit=10)
    assert any(l["action_type"] == "TEST_ACTION" for l in logs)
