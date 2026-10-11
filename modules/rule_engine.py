"""Inference Rule Engine supporting Forward and Backward Chaining with execution tracing."""
from typing import List, Dict, Any, Tuple, Optional
from modules.knowledge_base import KnowledgeBase, Rule, Action

class ForwardChainingEngine:
    """
    Data-driven forward reasoning engine.
    Starts with known initial facts, evaluates rule premises, and derives
    actionable recommendations with an auditable step-by-step trace.
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def infer(self, initial_facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes forward chaining on working memory facts.
        Returns triggered rules, recommended actions, and full execution trace.
        """
        working_memory = dict(initial_facts)
        fired_rules: List[Rule] = []
        recommendations: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []

        available_rules = list(self.kb.get_all_rules())
        fired_rule_ids = set()

        changed = True
        cycle = 1

        while changed and cycle <= 10:
            changed = False
            cycle_trace = {
                "cycle": cycle,
                "working_memory_snapshot": dict(working_memory),
                "evaluations": []
            }

            for rule in available_rules:
                if rule.rule_id in fired_rule_ids:
                    continue

                cond_evals = []
                all_matched = True
                for cond in rule.conditions:
                    matched = cond.evaluate(working_memory)
                    cond_evals.append({
                        "predicate": cond.predicate,
                        "operator": cond.operator,
                        "target_value": cond.target_value,
                        "actual_value": working_memory.get(cond.predicate, "<undefined>"),
                        "matched": matched
                    })
                    if not matched:
                        all_matched = False

                eval_record = {
                    "rule_id": rule.rule_id,
                    "title": rule.title,
                    "conditions": cond_evals,
                    "fired": all_matched
                }
                cycle_trace["evaluations"].append(eval_record)

                if all_matched:
                    fired_rules.append(rule)
                    fired_rule_ids.add(rule.rule_id)
                    recommendations.append({
                        "rule_id": rule.rule_id,
                        "title": rule.title,
                        "category": rule.action.category,
                        "recommendation": rule.action.recommendation,
                        "priority": rule.action.suggested_priority,
                        "target_team": rule.action.target_team,
                        "explanation": rule.action.explanation
                    })
                    # Add newly deduced facts if applicable
                    new_predicate = f"rule_fired_{rule.rule_id}"
                    working_memory[new_predicate] = True
                    changed = True

            trace.append(cycle_trace)
            cycle += 1

        return {
            "initial_facts": initial_facts,
            "final_working_memory": working_memory,
            "fired_rules": [r.rule_id for r in fired_rules],
            "recommendations": recommendations,
            "trace": trace,
            "success": len(fired_rules) > 0
        }

class BackwardChainingEngine:
    """
    Goal-driven backward reasoning engine.
    Starts with a target hypothesis/action and works backwards to verify
    whether available facts support or disprove the hypothesis.
    """
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def prove_hypothesis(self, target_rule_id: str, known_facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to prove whether conditions for a given rule can be satisfied
        given the available facts.
        """
        target_rule = self.kb.get_rule_by_id(target_rule_id)
        if not target_rule:
            return {
                "proven": False,
                "target_rule_id": target_rule_id,
                "error": f"Rule '{target_rule_id}' not found in Knowledge Base."
            }

        proof_steps: List[Dict[str, Any]] = []
        is_proven = True
        missing_facts: List[str] = []

        for cond in target_rule.conditions:
            predicate = cond.predicate
            has_fact = predicate in known_facts
            actual_val = known_facts.get(predicate, None)
            matched = cond.evaluate(known_facts) if has_fact else False

            proof_step = {
                "predicate": predicate,
                "required_operator": cond.operator,
                "required_target": cond.target_value,
                "fact_present": has_fact,
                "actual_value": actual_val,
                "satisfied": matched
            }
            proof_steps.append(proof_step)

            if not matched:
                is_proven = False
                if not has_fact:
                    missing_facts.append(predicate)

        return {
            "proven": is_proven,
            "target_rule_id": target_rule.rule_id,
            "rule_title": target_rule.title,
            "recommended_action": target_rule.action.recommendation if is_proven else None,
            "priority": target_rule.action.suggested_priority if is_proven else None,
            "proof_steps": proof_steps,
            "missing_facts": missing_facts,
            "explanation": (
                f"Hypothesis '{target_rule.title}' is fully SATISFIED by the available evidence."
                if is_proven else
                f"Hypothesis '{target_rule.title}' could NOT be proven. Unsatisfied conditions or missing facts: {missing_facts or 'Value mismatch'}."
            )
        }
