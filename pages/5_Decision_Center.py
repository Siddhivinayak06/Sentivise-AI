"""Page 5: Intelligent Decision Center - Core business workflow converting customer evidence into auditable action plans."""
import streamlit as st
import pandas as pd
from datetime import datetime

from services.decision_service import DecisionService
from database.repository import Repository
from config import THEME_COLORS, DEFAULT_PRIORITY_WEIGHTS
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_empty_state,
    render_section_header,
)
from components.sidebar import render_sidebar

st.set_page_config(page_title="Intelligent Decision Center", page_icon="⚖️", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Intelligent Decision Center & Issue Escalation",
    subtitle="Enterprise governance portal converting customer review evidence into prioritized business action items",
    badge="Core Business Value"
)

# Top Notice on Automation Safety
st.markdown(
    f"""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.8rem;">
        <span style="font-size: 1.3rem;">🛡️</span>
        <span style="font-size: 0.88rem; color: {THEME_COLORS['text_muted']};">
            <strong style="color: {THEME_COLORS['text']};">Safety & Governance Policy:</strong>
            Recommendations are assistive decision-support plans. The system will <strong>never</strong> automatically contact customers, issue financial refunds, or trigger product recalls without human sign-off.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Prioritization Weights Configuration Panel
with st.expander("⚙️ Configure Transparent Prioritization Scoring Weights", expanded=False):
    st.markdown(r"""
    $$\text{Priority Score} = w_f \cdot S_{\text{frequency}} + w_n \cdot S_{\text{neg\_ratio}} + w_s \cdot S_{\text{severity}} + w_r \cdot S_{\text{recency}} + w_p \cdot S_{\text{affected\_products}}$$
    *Scores are mapped to a 0–100 scale. Weights are normalized automatically.*
    """)
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        w_freq = st.slider("Complaint Frequency", 0.0, 1.0, DEFAULT_PRIORITY_WEIGHTS["frequency"], 0.05)
    with w2:
        w_neg = st.slider("Negative Ratio", 0.0, 1.0, DEFAULT_PRIORITY_WEIGHTS["negative_ratio"], 0.05)
    with w3:
        w_sev = st.slider("Severity Rating", 0.0, 1.0, DEFAULT_PRIORITY_WEIGHTS["severity"], 0.05)
    with w4:
        w_rec = st.slider("Recency Urgency", 0.0, 1.0, DEFAULT_PRIORITY_WEIGHTS["recency"], 0.05)
    with w5:
        w_prod = st.slider("Cross-Product Impact", 0.0, 1.0, DEFAULT_PRIORITY_WEIGHTS["affected_products"], 0.05)

    custom_weights = {
        "frequency": w_freq,
        "negative_ratio": w_neg,
        "severity": w_sev,
        "recency": w_rec,
        "affected_products": w_prod
    }

# Filter by Status
filter_c1, filter_c2, filter_c3 = st.columns([4, 4, 4])
with filter_c1:
    status_filter = st.selectbox("Filter Status", ["All", "Open", "Investigating", "Resolved", "Dismissed"], index=0)
with filter_c2:
    category_filter = st.selectbox("Filter Category", ["All", "Product Quality", "Logistics & Delivery", "Customer Support", "Packaging", "Pricing & Value"], index=0)
with filter_c3:
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    refresh_btn = st.button("🔄 Refresh Issues", use_container_width=True)

# Fetch issues from database
issues = DecisionService.list_issues(
    status_filter=status_filter if status_filter != "All" else None,
    category_filter=category_filter if category_filter != "All" else None
)

# Summary KPIs
k_open = sum(1 for i in issues if i["status"] == "Open")
k_inv = sum(1 for i in issues if i["status"] == "Investigating")
k_res = sum(1 for i in issues if i["status"] == "Resolved")
k_high = sum(1 for i in issues if i["priority_score"] >= 65)

kc1, kc2, kc3, kc4 = st.columns(4)
render_kpi(kc1, "Open Action Items", f"{k_open}", "Awaiting triage or assignment")
render_kpi(kc2, "In Active Investigation", f"{k_inv}", "Assigned to departmental owners")
render_kpi(kc3, "Resolved & Verified", f"{k_res}", "Historical resolution archived")
render_kpi(kc4, "High Priority (>=65)", f"{k_high}", "Requiring immediate operational review")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# New Issue Modal / Form
with st.expander("➕ Log New Customer Issue Manually", expanded=False):
    with st.form("new_issue_form"):
        ni_title = st.text_input("Issue Title", placeholder="e.g. Firmware Freeze on 1.0.4 Update")
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            ni_category = st.selectbox("Category", ["Product Quality", "Logistics & Delivery", "Customer Support", "Packaging", "Pricing & Value", "Safety & Compliance"])
        with c_b:
            ni_severity = st.selectbox("Severity", ["Critical", "High", "Medium", "Low"], index=1)
        with c_c:
            ni_freq = st.number_input("Supporting Reviews Count", min_value=1, value=5)
        
        c_d, c_e = st.columns(2)
        with c_d:
            ni_neg_ratio = st.slider("Negative Sentiment Ratio", 0.0, 1.0, 0.85, 0.05)
        with c_e:
            ni_prod_cnt = st.number_input("Affected SKUs / Products Count", min_value=1, value=1)
            
        ni_action = st.text_area("Recommended Action", placeholder="e.g. Roll back firmware OTA and audit regression test suite.")
        submit_issue = st.form_submit_button("Create Tracked Business Issue", type="primary")
        
        if submit_issue:
            if not ni_title.strip():
                st.error("Issue title cannot be blank.")
            else:
                new_id = DecisionService.create_new_issue(
                    title=ni_title,
                    category=ni_category,
                    severity=ni_severity,
                    frequency=int(ni_freq),
                    negative_ratio=float(ni_neg_ratio),
                    affected_products_count=int(ni_prod_cnt),
                    recommended_action=ni_action,
                    weights=custom_weights
                )
                st.success(f"Created tracked issue #{new_id}!")
                st.rerun()

# List Issues
if not issues:
    render_empty_state("No Tracked Issues Found", "Create a new issue above or run batch analytics to discover candidate problems.")
else:
    render_section_header("📋", "Active Issue Queue")
    for issue in issues:
        issue_id = issue["id"]
        status = issue["status"]
        sev = issue["severity"]
        score = issue["priority_score"]
        
        # Status chip class
        status_cls = "status-open"
        if status == "Investigating":
            status_cls = "status-investigating"
        elif status == "Resolved":
            status_cls = "status-resolved"
        
        with st.container():
            st.markdown(
                f"""
                <div class="issue-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.6rem;">
                        <div style="display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center;">
                            <span class="badge badge-info">#{issue_id} {issue['category']}</span>
                            <span class="badge badge-warning">Severity: {sev}</span>
                            <span class="status-chip {status_cls}">
                                <span class="pulse-dot {"pulse-dot-green" if status == "Resolved" else "pulse-dot-red" if status == "Open" else "pulse-dot-blue"}"></span>
                                {status}
                            </span>
                        </div>
                        <div style="text-align: right; flex-shrink: 0;">
                            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {THEME_COLORS['text_muted']};">Score</div>
                            <div class="score-ring">{score:.0f}</div>
                        </div>
                    </div>
                    <h3 style="font-size: 1.15rem; font-weight: 700; color: {THEME_COLORS['text']}; margin: 0.3rem 0 0.5rem 0;">{issue['title']}</h3>
                    <p style="font-size: 0.9rem; color: {THEME_COLORS['text']}; margin-bottom: 0.4rem;">
                        <strong style="color: {THEME_COLORS['primary']};">💡 Action:</strong> {issue['recommended_action']}
                    </p>
                    <p style="font-size: 0.82rem; color: {THEME_COLORS['text_muted']}; margin-bottom: 0.5rem;">
                        📊 <strong>Rationale:</strong> {issue['rationale']}
                    </p>
                    <div style="font-size: 0.75rem; color: {THEME_COLORS['text_muted']}; display: flex; flex-wrap: wrap; gap: 1.25rem;">
                        <span>👤 Owner: <strong>{issue['assigned_to']}</strong></span>
                        <span>📅 First Seen: {issue['first_seen'] or 'N/A'}</span>
                        <span>📅 Last Seen: {issue['last_seen'] or 'N/A'}</span>
                        <span>📦 Reviews: {issue['supporting_reviews_count']}</span>
                        <span>🏷️ Products: {issue['affected_products_count']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Interactive Controls for this issue
            with st.expander(f"🛠️ Manage Issue #{issue_id} (Status, Assignee & Action Audit)", expanded=False):
                col_act1, col_act2, col_act3 = st.columns(3)
                with col_act1:
                    new_status_val = st.selectbox(
                        "Update Status",
                        ["Open", "Investigating", "Resolved", "Dismissed"],
                        index=["Open", "Investigating", "Resolved", "Dismissed"].index(status),
                        key=f"status_sel_{issue_id}"
                    )
                with col_act2:
                    new_owner_val = st.text_input("Assigned Owner", value=issue["assigned_to"], key=f"owner_{issue_id}")
                with col_act3:
                    action_note = st.text_input("Resolution Note / Audit Reason", placeholder="e.g. Root cause verified with vendor", key=f"note_{issue_id}")
                
                if st.button("💾 Apply Status & Assignment Change", key=f"btn_update_{issue_id}", type="primary"):
                    ok = DecisionService.update_status(
                        issue_id=issue_id,
                        new_status=new_status_val,
                        assigned_to=new_owner_val,
                        author="Business Analyst",
                        note=action_note
                    )
                    if ok:
                        st.success(f"Issue #{issue_id} updated to '{new_status_val}'.")
                        st.rerun()

                # Display Note Timeline
                notes = Repository.get_issue_notes(issue_id)
                if notes:
                    st.markdown("##### 📜 Historical Audit Notes")
                    for n in notes:
                        st.caption(f"**{n['author']}** ({n['created_at']}): {n['note']}")
