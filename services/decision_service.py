"""Decision Center Service coordinating issue identification, prioritization, and lifecycle management."""
from typing import Dict, Any, List, Optional
from datetime import datetime
from modules.issue_prioritization import IssuePrioritizer
from database.repository import Repository
from config import DEFAULT_PRIORITY_WEIGHTS

class DecisionService:
    @staticmethod
    def get_prioritizer(weights: Optional[Dict[str, float]] = None) -> IssuePrioritizer:
        return IssuePrioritizer(weights or DEFAULT_PRIORITY_WEIGHTS)

    @staticmethod
    def list_issues(status_filter: Optional[str] = None, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches all issues from SQLite repository."""
        return Repository.get_issues(status=status_filter, category=category_filter)

    @staticmethod
    def get_issue_details(issue_id: int) -> Optional[Dict[str, Any]]:
        """Returns issue record and note timeline."""
        issue = Repository.get_issue_by_id(issue_id)
        if not issue:
            return None
        notes = Repository.get_issue_notes(issue_id)
        issue["notes"] = notes
        return issue

    @staticmethod
    def update_status(
        issue_id: int,
        new_status: str,
        assigned_to: Optional[str] = None,
        author: str = "Decision Center Manager",
        note: Optional[str] = None
    ) -> bool:
        """Transitions issue lifecycle state and records audit log."""
        return Repository.update_issue_status(
            issue_id=issue_id,
            new_status=new_status,
            assigned_to=assigned_to,
            author=author,
            note=note
        )

    @staticmethod
    def add_note(issue_id: int, author: str, note: str) -> int:
        return Repository.add_issue_note(issue_id, author, note)

    @staticmethod
    def create_new_issue(
        title: str,
        category: str,
        severity: str,
        frequency: int,
        negative_ratio: float,
        affected_products_count: int = 1,
        recency_days: int = 7,
        recommended_action: str = "",
        rule_identifiers: str = "",
        first_seen: Optional[str] = None,
        last_seen: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> int:
        """Calculates transparent priority score and creates an issue record."""
        prioritizer = DecisionService.get_prioritizer(weights)
        score, breakdown, rationale = prioritizer.calculate_priority(
            frequency=frequency,
            negative_ratio=negative_ratio,
            severity=severity,
            recency_days=recency_days,
            affected_products=affected_products_count
        )
        return Repository.create_issue(
            title=title,
            category=category,
            severity=severity,
            status="Open",
            assigned_to="Unassigned",
            priority_score=score,
            rationale=rationale,
            recommended_action=recommended_action,
            rule_identifiers=rule_identifiers,
            affected_products_count=affected_products_count,
            supporting_reviews_count=frequency,
            first_seen=first_seen or datetime.now().strftime("%Y-%m-%d"),
            last_seen=last_seen or datetime.now().strftime("%Y-%m-%d")
        )

    @staticmethod
    def seed_initial_issues_if_needed():
        """Populates initial realistic business issues if database is empty."""
        existing = Repository.get_issues()
        if len(existing) > 0:
            return

        demo_issues = [
            {
                "title": "Severe Battery Degradation & Overheating in Audio Units",
                "category": "Product Quality",
                "severity": "High",
                "frequency": 42,
                "negative_ratio": 0.88,
                "affected_products_count": 3,
                "recency_days": 2,
                "recommended_action": "Trigger Hardware Engineering audit on lithium-ion cell thermal throttling and vendor batch #440.",
                "rule_identifiers": "RULE_QUAL_01, RULE_SAFETY_01"
            },
            {
                "title": "East Coast 3PL Delivery Delays and Lost Transit Parcels",
                "category": "Logistics & Delivery",
                "severity": "Medium",
                "frequency": 28,
                "negative_ratio": 0.75,
                "affected_products_count": 5,
                "recency_days": 4,
                "recommended_action": "Audit Regional 3PL delivery SLA performance and institute automated dispatch tracking.",
                "rule_identifiers": "RULE_LOGIS_01"
            },
            {
                "title": "Unresponsive Tier-1 Customer Support & Long Chat Queues",
                "category": "Customer Support",
                "severity": "High",
                "frequency": 19,
                "negative_ratio": 0.95,
                "affected_products_count": 2,
                "recency_days": 1,
                "recommended_action": "Deploy additional Tier-1 helpdesk shifts and review ticketing SLA violations.",
                "rule_identifiers": "RULE_CS_01"
            },
            {
                "title": "Crushed Retail Packaging During Bulk Transit",
                "category": "Packaging",
                "severity": "Low",
                "frequency": 12,
                "negative_ratio": 0.60,
                "affected_products_count": 1,
                "recency_days": 10,
                "recommended_action": "Reinforce cardboard gauge for exterior master cartons at assembly center.",
                "rule_identifiers": "RULE_PACK_01"
            }
        ]

        for item in demo_issues:
            DecisionService.create_new_issue(
                title=item["title"],
                category=item["category"],
                severity=item["severity"],
                frequency=item["frequency"],
                negative_ratio=item["negative_ratio"],
                affected_products_count=item["affected_products_count"],
                recency_days=item["recency_days"],
                recommended_action=item["recommended_action"],
                rule_identifiers=item["rule_identifiers"]
            )
