"""Unit tests for the Knowledge Base and Forward/Backward Chaining Rule Engines."""
import pytest
from modules.knowledge_base import KnowledgeBase, Condition
from modules.rule_engine import ForwardChainingEngine, BackwardChainingEngine

def test_knowledge_base_rule_catalog():
    kb = KnowledgeBase()
    rules = kb.get_all_rules()
    assert len(rules) >= 10
    rule_ids = [r.rule_id for r in rules]
    assert "RULE_SAFETY_01" in rule_ids
    assert "RULE_QUAL_01" in rule_ids
    assert "RULE_LOGIS_01" in rule_ids

def test_condition_evaluation():
    c1 = Condition("sentiment", "==", "Negative")
    assert c1.evaluate({"sentiment": "Negative"})
    assert not c1.evaluate({"sentiment": "Positive"})

    c2 = Condition("confidence", ">=", 0.80)
    assert c2.evaluate({"confidence": 0.85})
    assert not c2.evaluate({"confidence": 0.75})

def test_forward_chaining_safety_rule():
    kb = KnowledgeBase()
    engine = ForwardChainingEngine(kb)
    facts = {
        "sentiment": "Negative",
        "safety_hazard_detected": True,
        "confidence": 0.95
    }
    result = engine.infer(facts)
    assert result["success"]
    assert "RULE_SAFETY_01" in result["fired_rules"]
    assert any("Product Safety" in r["target_team"] for r in result["recommendations"])

def test_forward_chaining_no_match_returns_empty():
    kb = KnowledgeBase()
    engine = ForwardChainingEngine(kb)
    facts = {
        "sentiment": "Positive",
        "confidence": 0.75  # Standard positive confidence: neither uncertainty (<0.60) nor champion (>=0.90) fires
    }
    result = engine.infer(facts)
    assert not result["success"]
    assert len(result["fired_rules"]) == 0

def test_backward_chaining_proof():
    kb = KnowledgeBase()
    engine = BackwardChainingEngine(kb)

    # Proving durability defect rule
    facts = {
        "sentiment": "Negative",
        "aspect_quality": True,
        "confidence": 0.75
    }
    proof = engine.prove_hypothesis("RULE_QUAL_01", facts)
    assert proof["proven"]
    assert "Hardware Engineering" in proof["recommended_action"]

    # Failing proof when condition is missing
    incomplete_facts = {
        "sentiment": "Positive", # Contradicts Negative requirement
        "aspect_quality": True,
        "confidence": 0.75
    }
    proof_failed = engine.prove_hypothesis("RULE_QUAL_01", incomplete_facts)
    assert not proof_failed["proven"]
