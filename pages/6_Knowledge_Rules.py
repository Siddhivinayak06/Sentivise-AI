"""Page 6: Knowledge Base and Rule Engine demonstrating forward and backward chaining reasoning."""
import streamlit as st
import pandas as pd
from modules.knowledge_base import KnowledgeBase
from modules.rule_engine import ForwardChainingEngine, BackwardChainingEngine
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.sidebar import render_sidebar
from config import THEME_COLORS

st.set_page_config(page_title="Knowledge Base & Rules", page_icon="🧠", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Knowledge Base & Expert Rule Reasoning Engine",
    subtitle="Inspect formal domain rules, execute forward and backward chaining inference, and view auditable proof traces",
    badge="Symbolic Reasoning"
)

kb = KnowledgeBase()
all_rules = kb.get_all_rules()

tab_rules, tab_forward, tab_backward, tab_test = st.tabs([
    "📖 Rule Catalog & Facts Explorer",
    "➡️ Forward Chaining Inference",
    "⬅️ Backward Chaining (Goal Proving)",
    "🧪 Test Cases & Validation"
])

# Tab 1: Rule Catalog
with tab_rules:
    render_section_header("📚", "Enterprise Decision Rules Catalog")
    st.caption(f"The Knowledge Base contains **{len(all_rules)} verified business rules** separating ML sentiment prediction from business actions.")

    rule_rows = []
    for r in all_rules:
        cond_str = " AND ".join([f"{c.predicate} {c.operator} {c.target_value}" for c in r.conditions])
        rule_rows.append({
            "Rule ID": r.rule_id,
            "Title": r.title,
            "Category": r.category,
            "IF Conditions": cond_str,
            "THEN Action": r.action.recommendation,
            "Priority": r.action.suggested_priority,
            "Target Team": r.action.target_team
        })

    rule_df = pd.DataFrame(rule_rows)
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

# Tab 2: Forward Chaining
with tab_forward:
    render_section_header("➡️", "Forward Chaining Reasoning Engine")
    st.caption("Data-driven deduction: Supply input facts, and the engine iterates through rule premises to deduce actions.")

    fc_col1, fc_col2 = st.columns([5, 7])

    with fc_col1:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
                border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 1.05rem; margin-bottom: 0.8rem;">
                    1. Define Working Memory Facts
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        f_sent = st.selectbox("Sentiment", ["Negative", "Positive"], index=0)
        f_conf = st.slider("Prediction Confidence", 0.0, 1.0, 0.85, 0.05)
        f_qual = st.checkbox("aspect_quality (Hardware/defect mentioned)", value=True)
        f_logis = st.checkbox("aspect_delivery (Transit/shipping delay)", value=False)
        f_pack = st.checkbox("aspect_packaging (Damaged box/wrap)", value=False)
        f_price = st.checkbox("aspect_price (Price/value complaint)", value=False)
        f_supp = st.checkbox("aspect_support (Unhelpful agent/hotline)", value=False)
        f_safety = st.checkbox("safety_hazard_detected (Fire/burn/hazard)", value=False)
        f_fraud = st.checkbox("counterfeit_risk (Fake/replica token)", value=False)

        input_facts = {
            "sentiment": f_sent,
            "confidence": f_conf,
            "aspect_quality": f_qual,
            "aspect_delivery": f_logis,
            "aspect_packaging": f_pack,
            "aspect_price": f_price,
            "aspect_support": f_supp,
            "safety_hazard_detected": f_safety,
            "counterfeit_risk": f_fraud,
            "high_severity": bool(f_safety or (f_conf >= 0.85 and f_sent == "Negative"))
        }

        run_fc_btn = st.button("🚀 Run Forward Chaining", type="primary", use_container_width=True)

    with fc_col2:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
                border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 1.05rem; margin-bottom: 0.8rem;">
                    2. Inference Results & Audit Trace
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if run_fc_btn:
            engine = ForwardChainingEngine(kb)
            fc_output = engine.infer(input_facts)

            if not fc_output["success"]:
                st.warning("⚠️ No rules fired for the given facts. The system honestly reports that no specialized recommendation is justified.")
            else:
                st.success(f"Fired {len(fc_output['fired_rules'])} Rules: {', '.join(fc_output['fired_rules'])}")
                for rec in fc_output["recommendations"]:
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                                <strong style="color: {THEME_COLORS['text']}; font-size: 1.02rem;">{rec['title']} ({rec['rule_id']})</strong>
                                <span class="badge badge-warning">{rec['priority']} Priority</span>
                            </div>
                            <p style="margin: 0.4rem 0; font-weight: 600; color: {THEME_COLORS['secondary']}; font-size: 0.95rem;">
                                Action: {rec['recommendation']}
                            </p>
                            <p style="margin: 0; font-size: 0.85rem; color: {THEME_COLORS['text_muted']};">
                                Target Department: <strong style="color: {THEME_COLORS['text']};">{rec['target_team']}</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with st.expander("📜 Step-by-Step Forward Chaining Trace", expanded=True):
                for cycle in fc_output["trace"]:
                    st.markdown(f"**Cycle {cycle['cycle']} Evaluations:**")
                    for ev in cycle["evaluations"]:
                        icon = "✅" if ev["fired"] else "❌"
                        st.write(f"{icon} **{ev['rule_id']} ({ev['title']})**: Fired={ev['fired']}")

# Tab 3: Backward Chaining
with tab_backward:
    render_section_header("⬅️", "Backward Chaining (Goal-Driven Proof)")
    st.caption("Hypothesis validation: Pick a candidate target business rule, and the engine evaluates whether evidence supports or disproves it.")

    bc_col1, bc_col2 = st.columns([5, 7])

    with bc_col1:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
                border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 1.05rem; margin-bottom: 0.8rem;">
                    1. Select Hypothesis Rule
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        rule_choices = {f"{r.rule_id} - {r.title}": r.rule_id for r in all_rules}
        sel_rule_label = st.selectbox("Hypothesis Rule to Prove", list(rule_choices.keys()))
        target_rule_id = rule_choices[sel_rule_label]

        st.markdown("#### 2. Available Facts for Verification")
        b_sent = st.selectbox("Observed Sentiment", ["Negative", "Positive"], index=0, key="bc_sent")
        b_conf = st.slider("Observed Confidence", 0.0, 1.0, 0.75, 0.05, key="bc_conf")
        b_qual = st.checkbox("Fact: aspect_quality is True", value=True, key="bc_qual")
        b_safe = st.checkbox("Fact: safety_hazard_detected is True", value=False, key="bc_safe")
        b_del = st.checkbox("Fact: aspect_delivery is True", value=False, key="bc_del")

        known_facts = {
            "sentiment": b_sent,
            "confidence": b_conf,
            "aspect_quality": b_qual,
            "safety_hazard_detected": b_safe,
            "aspect_delivery": b_del,
            "high_severity": bool(b_safe or (b_conf >= 0.85 and b_sent == "Negative"))
        }

        run_bc_btn = st.button("🔍 Prove Hypothesis", type="primary", use_container_width=True)

    with bc_col2:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {THEME_COLORS['border']};
                border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <div style="font-weight: 700; color: {THEME_COLORS['text']}; font-size: 1.05rem; margin-bottom: 0.8rem;">
                    3. Hypothesis Verification & Proof Tree
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if run_bc_btn:
            b_engine = BackwardChainingEngine(kb)
            proof = b_engine.prove_hypothesis(target_rule_id, known_facts)

            if proof["proven"]:
                st.success(f"✅ Hypothesis PROVEN: {proof['rule_title']}")
                st.markdown(f"**Recommended Action**: `{proof['recommended_action']}`")
            else:
                st.error(f"❌ Hypothesis NOT PROVEN: {proof['rule_title']}")
                st.write(proof["explanation"])

            st.markdown("##### 🌲 Condition Proof Steps:")
            steps_df = pd.DataFrame(proof["proof_steps"])
            st.dataframe(steps_df, use_container_width=True, hide_index=True)

# Tab 4: Test Cases
with tab_test:
    render_section_header("🧪", "Rule Engine Integration Test Suite")
    st.caption("Preset integration test scenarios validating rule evaluation correctness against ground-truth conditions:")

    test_scenarios = [
        {
            "name": "Scenario 1: Critical Safety Hazard",
            "facts": {"sentiment": "Negative", "safety_hazard_detected": True, "confidence": 0.95},
            "expected_rule": "RULE_SAFETY_01"
        },
        {
            "name": "Scenario 2: Hardware Durability Defect",
            "facts": {"sentiment": "Negative", "aspect_quality": True, "confidence": 0.70},
            "expected_rule": "RULE_QUAL_01"
        },
        {
            "name": "Scenario 3: Delivery Delay",
            "facts": {"sentiment": "Negative", "aspect_delivery": True, "confidence": 0.80},
            "expected_rule": "RULE_LOGIS_01"
        },
        {
            "name": "Scenario 4: High Confidence Promoter",
            "facts": {"sentiment": "Positive", "confidence": 0.95},
            "expected_rule": "RULE_POS_CHAMP_01"
        }
    ]

    for tc in test_scenarios:
        with st.expander(f"Test: {tc['name']}", expanded=True):
            engine = ForwardChainingEngine(kb)
            out = engine.infer(tc["facts"])
            passed = tc["expected_rule"] in out["fired_rules"]
            icon = "✅ PASSED" if passed else "❌ FAILED"
            st.markdown(f"**Result:** `{icon}` | **Fired Rules:** `{out['fired_rules']}` (Expected: `{tc['expected_rule']}`)")
