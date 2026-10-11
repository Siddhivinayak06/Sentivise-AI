"""Knowledge Base Module defining entities, facts, and business decision rules."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

@dataclass
class Fact:
    predicate: str
    value: Any
    confidence: float = 1.0
    source: str = "Input"

    def __repr__(self):
        return f"Fact({self.predicate}={self.value}, conf={self.confidence:.2f})"

@dataclass
class Condition:
    predicate: str
    operator: str  # '==', '!=', '>', '<', '>=', '<=', 'contains', 'in'
    target_value: Any

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        if self.predicate not in facts:
            return False
        val = facts[self.predicate]
        if self.operator == '==':
            return val == self.target_value
        elif self.operator == '!=':
            return val != self.target_value
        elif self.operator == '>':
            return float(val) > float(self.target_value)
        elif self.operator == '<':
            return float(val) < float(self.target_value)
        elif self.operator == '>=':
            return float(val) >= float(self.target_value)
        elif self.operator == '<=':
            return float(val) <= float(self.target_value)
        elif self.operator == 'contains':
            return str(self.target_value).lower() in str(val).lower()
        elif self.operator == 'in':
            return val in self.target_value
        return False

@dataclass
class Action:
    recommendation: str
    category: str
    suggested_priority: str
    target_team: str
    explanation: str

@dataclass
class Rule:
    rule_id: str
    title: str
    description: str
    category: str
    conditions: List[Condition]
    action: Action
    enabled: bool = True

    def matches(self, facts: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        return all(cond.evaluate(facts) for cond in self.conditions)

class KnowledgeBase:
    """Central repository of verified domain knowledge and expert decision rules."""

    def __init__(self):
        self.rules: List[Rule] = self._load_default_rules()

    def get_all_rules(self) -> List[Rule]:
        return self.rules

    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        return None

    def get_rules_by_category(self, category: str) -> List[Rule]:
        return [r for r in self.rules if r.category.lower() == category.lower()]

    def _load_default_rules(self) -> List[Rule]:
        return [
            Rule(
                rule_id="RULE_SAFETY_01",
                title="Critical Safety Hazard Alert",
                description="IF sentiment is negative and review mentions critical safety risks (fire, burn, smoke, explosion, choke, shock), THEN immediately flag for urgent safety compliance audit and executive recall review.",
                category="Safety & Compliance",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("safety_hazard_detected", "==", True)
                ],
                action=Action(
                    recommendation="IMMEDIATE ESCALATION: Quarantine affected batch and initiate urgent Product Safety Compliance review.",
                    category="Safety & Compliance",
                    suggested_priority="Critical",
                    target_team="Product Safety & Legal",
                    explanation="Review contains high-severity safety risk markers requiring immediate statutory compliance evaluation."
                )
            ),
            Rule(
                rule_id="RULE_QUAL_01",
                title="Repeated Durability Defect Escalation",
                description="IF sentiment is negative and physical defect/durability issues are reported, THEN trigger hardware engineering investigation.",
                category="Product Quality",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("aspect_quality", "==", True),
                    Condition("confidence", ">=", 0.60)
                ],
                action=Action(
                    recommendation="Trigger Hardware Engineering & QA audit on component durability and assembly tolerances.",
                    category="Product Quality",
                    suggested_priority="High",
                    target_team="Engineering & QA",
                    explanation="Negative customer feedback corroborated by defect/breakage lexical markers."
                )
            ),
            Rule(
                rule_id="RULE_LOGIS_01",
                title="Carrier Delivery Performance Delay",
                description="IF sentiment is negative and complaints mention shipping or delivery delays, THEN audit logistics service level agreements.",
                category="Logistics & Delivery",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("aspect_delivery", "==", True)
                ],
                action=Action(
                    recommendation="Review 3PL courier transit times and initiate logistics partner SLA audit.",
                    category="Logistics & Delivery",
                    suggested_priority="Medium",
                    target_team="Logistics & Fulfillment",
                    explanation="Customer reported transit delays or courier delivery failures."
                )
            ),
            Rule(
                rule_id="RULE_PACK_01",
                title="Transit Packaging Damage Warning",
                description="IF sentiment is negative and packaging is reported torn, smashed, or missing protective wrap, THEN review packaging engineering.",
                category="Packaging",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("aspect_packaging", "==", True)
                ],
                action=Action(
                    recommendation="Inspect warehouse packaging specifications and reinforce protective cushioning for transit.",
                    category="Packaging",
                    suggested_priority="Medium",
                    target_team="Warehouse Packaging Operations",
                    explanation="Item arrived damaged or in compromised packaging."
                )
            ),
            Rule(
                rule_id="RULE_PRIC_01",
                title="Price-to-Value Perception Gap",
                description="IF sentiment is negative and customers report the product is overpriced or poor value, THEN evaluate pricing strategy.",
                category="Pricing & Value",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("aspect_price", "==", True)
                ],
                action=Action(
                    recommendation="Conduct competitive pricing benchmarking and consider value-bundle promotions.",
                    category="Pricing & Value",
                    suggested_priority="Low",
                    target_team="Product Marketing & Pricing",
                    explanation="Customer perceives feature set as insufficient for the current price point."
                )
            ),
            Rule(
                rule_id="RULE_CS_01",
                title="Customer Support Escalation & Responsiveness",
                description="IF sentiment is negative and dissatisfaction is directed at customer support agents or response delays, THEN escalate to support operations.",
                category="Customer Support",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("aspect_support", "==", True)
                ],
                action=Action(
                    recommendation="Escalate customer case to Tier-2 senior resolution team and audit helpdesk response times.",
                    category="Customer Support",
                    suggested_priority="High",
                    target_team="Customer Experience Operations",
                    explanation="Customer reported unhelpful, rude, or unresponsive customer service."
                )
            ),
            Rule(
                rule_id="RULE_USA_01",
                title="Usability & Setup Guidance Defect",
                description="IF sentiment is negative and customer struggled with onboarding, setup, or confusing instructions, THEN revise product onboarding.",
                category="Usability & CX",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("aspect_usability", "==", True)
                ],
                action=Action(
                    recommendation="Revise quick-start setup guide and publish instructional onboarding video on product page.",
                    category="Usability & CX",
                    suggested_priority="Medium",
                    target_team="UX Design & Technical Documentation",
                    explanation="Customer experienced usability barriers or unclear user manual instructions."
                )
            ),
            Rule(
                rule_id="RULE_VIP_01",
                title="High-Dissatisfaction Churn Intervention",
                description="IF rating is 1-star (or strong negative confidence > 0.85) AND multiple complaints are expressed, THEN flag for proactive VIP retention.",
                category="Customer Retention",
                conditions=[
                    Condition("sentiment", "==", "Negative"),
                    Condition("confidence", ">=", 0.85),
                    Condition("high_severity", "==", True)
                ],
                action=Action(
                    recommendation="Dispatch dedicated retention agent with replacement voucher or courtesy follow-up call.",
                    category="Customer Retention",
                    suggested_priority="High",
                    target_team="Customer Retention & VIP Desk",
                    explanation="Extremely high negative model confidence combined with severe dissatisfaction indicates high churn risk."
                )
            ),
            Rule(
                rule_id="RULE_UNCERT_01",
                title="Low Confidence / Ambiguous Review Flag",
                description="IF model prediction confidence is low (< 0.60) or features are contradictory, THEN route for manual human review.",
                category="Data Governance & Verification",
                conditions=[
                    Condition("confidence", "<", 0.60)
                ],
                action=Action(
                    recommendation="Route to human moderation queue for manual verification before automated escalation.",
                    category="Data Governance & Verification",
                    suggested_priority="Low",
                    target_team="Moderation & Quality Assurance",
                    explanation="Prediction confidence is below decision threshold (uncertain prediction)."
                )
            ),
            Rule(
                rule_id="RULE_POS_CHAMP_01",
                title="Brand Champion Testimonial Candidate",
                description="IF sentiment is strongly positive (confidence >= 0.90) and features praise quality/service, THEN recommend for marketing showcase.",
                category="Marketing & Advocacy",
                conditions=[
                    Condition("sentiment", "==", "Positive"),
                    Condition("confidence", ">=", 0.90)
                ],
                action=Action(
                    recommendation="Invite customer to join Brand Ambassador program and request permission for featured homepage testimonial.",
                    category="Marketing & Advocacy",
                    suggested_priority="Low",
                    target_team="Marketing & Growth",
                    explanation="High positive sentiment confidence indicates an enthusiastic brand promoter."
                )
            ),
            Rule(
                rule_id="RULE_FRAUD_01",
                title="Counterfeit / Authenticity Dispute Warning",
                description="IF review mentions fake, replica, knockoff, or counterfeit goods, THEN flag for Marketplace Brand Protection inspection.",
                category="Marketplace Integrity",
                conditions=[
                    Condition("counterfeit_risk", "==", True)
                ],
                action=Action(
                    recommendation="Inspect seller inventory lot against manufacturer authenticity seals and verify supply chain chain of custody.",
                    category="Marketplace Integrity",
                    suggested_priority="Critical",
                    target_team="Brand Protection & Compliance",
                    explanation="Customer suspected receiving non-genuine or counterfeit product."
                )
            ),
            Rule(
                rule_id="RULE_RESOLV_01",
                title="Post-Resolution Quality Verification",
                description="IF issue has been addressed by engineering, THEN archive historical resolution and log audit record.",
                category="Resolution Governance",
                conditions=[
                    Condition("status", "==", "Resolved")
                ],
                action=Action(
                    recommendation="Retain verified audit evidence and monitor post-resolution return telemetry for 30 days.",
                    category="Resolution Governance",
                    suggested_priority="Low",
                    target_team="Operations Management",
                    explanation="Issue marked resolved; ongoing monitoring confirmed."
                )
            )
        ]
