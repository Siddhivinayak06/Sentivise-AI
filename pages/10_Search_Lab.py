"""Page 10: Search Algorithms Lab evaluating BFS, DFS, and A* on a structured customer resolution state space."""
import streamlit as st
import pandas as pd
from services.search_service import SearchService
from components.ui_elements import (
    inject_custom_css,
    render_header,
    render_kpi,
    render_section_header,
)
from components.sidebar import render_sidebar
from config import THEME_COLORS

st.set_page_config(page_title="Search Algorithms Lab", page_icon="🗺️", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Search Algorithms & State-Space Planning Lab",
    subtitle="Comparative exploration of Uninformed (BFS, DFS) and Informed (A*) search algorithms for issue escalation workflows",
    badge="Algorithmic Foundations"
)

st.markdown(
    f"""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.8rem;">
        <span style="font-size: 1.3rem;">ℹ️</span>
        <span style="font-size: 0.88rem; color: {THEME_COLORS['text_muted']};">
            <strong style="color: {THEME_COLORS['text']};">Academic Integration Context:</strong>
            This module implements <strong>Automated Planning & Graph Search</strong> algorithms on a discrete operational workflow graph.
            BFS, DFS, and A* operate as discrete planning techniques independent from the continuous sentiment classifiers.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Available states
states = SearchService.get_available_states()

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    start_node = st.selectbox("Start State Node", states, index=0)
with col_ctrl2:
    goal_node = st.selectbox("Goal State Node", states, index=len(states) - 1)
with col_ctrl3:
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    run_search_btn = st.button("🚀 Execute Comparative Search", type="primary", use_container_width=True)

# Run Comparative Search
search_res = SearchService.compare_search_algorithms(start_node, goal_node)
comp_table = search_res["comparison"]

# Benchmarking Table
render_section_header("📊", "Search Algorithm Performance Comparison")
st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

tab_astar, tab_bfs, tab_dfs, tab_graph = st.tabs([
    "⭐ A* Informed Search",
    "🌊 Breadth-First Search (BFS)",
    "🌲 Depth-First Search (DFS)",
    "🕸️ Graph Structure & Admissible Heuristics"
])

def render_path_chips(path_nodes):
    chips_html = "".join([
        f'<span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35); color: {THEME_COLORS["secondary"]}; font-weight: 700; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.88rem;">{node}</span>'
        + (' <span style="color: ' + THEME_COLORS['primary'] + '; font-weight: 800; font-size: 1.1rem; margin: 0 0.2rem;">&rarr;</span> ' if idx < len(path_nodes) - 1 else '')
        for idx, node in enumerate(path_nodes)
    ])
    st.markdown(
        f'<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid {THEME_COLORS["border"]}; border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0; display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem;">{chips_html}</div>',
        unsafe_allow_html=True
    )

# Tab 1: A*
with tab_astar:
    a_data = search_res["astar"]
    render_section_header("⭐", "A* Informed Search Results")
    st.caption("Evaluates f(n) = g(n) + h(n) utilizing an admissible heuristic measuring estimated hours to resolution.")

    ac1, ac2, ac3, ac4 = st.columns(4)
    render_kpi(ac1, "Path Found", "Yes" if a_data["success"] else "No", "Goal reached")
    render_kpi(ac2, "Optimal Path Cost", f"{a_data['path_cost']} hrs", "Sum of edge transitions")
    render_kpi(ac3, "Nodes Expanded", f"{a_data['expanded_nodes_count']}", "Visited search states")
    render_kpi(ac4, "Runtime", f"{a_data['execution_time_ms']} ms", "Microsecond execution")

    st.markdown("##### 📍 Optimal Resolution Sequence:")
    render_path_chips(a_data["path"])

    with st.expander("Order of Node Expansion"):
        st.write(" &rarr; ".join(a_data["expanded_nodes"]))

# Tab 2: BFS
with tab_bfs:
    b_data = search_res["bfs"]
    render_section_header("🌊", "Breadth-First Search (BFS)")
    st.caption("FIFO Queue traversal: Guarantees shortest path in number of transitions (hops).")

    bc1, bc2, bc3 = st.columns(3)
    render_kpi(bc1, "Path Length (Hops)", f"{len(b_data['path'])} states", "Step transitions")
    render_kpi(bc2, "Accumulated Cost", f"{b_data['path_cost']} hrs", "Total labor hours")
    render_kpi(bc3, "Expanded Nodes", f"{b_data['expanded_nodes_count']}", "Exhaustive level search")

    st.markdown("##### 📍 BFS Discovered Path:")
    render_path_chips(b_data["path"])

# Tab 3: DFS
with tab_dfs:
    d_data = search_res["dfs"]
    render_section_header("🌲", "Depth-First Search (DFS)")
    st.caption("LIFO Stack traversal: Deeply explores branch paths before backtracking.")

    dc1, dc2, dc3 = st.columns(3)
    render_kpi(dc1, "Path Length (Hops)", f"{len(d_data['path'])} states", "Branch path")
    render_kpi(dc2, "Accumulated Cost", f"{d_data['path_cost']} hrs", "Total labor hours")
    render_kpi(dc3, "Expanded Nodes", f"{d_data['expanded_nodes_count']}", "Branch search")

    st.markdown("##### 📍 DFS Discovered Path:")
    render_path_chips(d_data["path"])

# Tab 4: Graph Structure
with tab_graph:
    render_section_header("🕸️", "Workflow Graph Representation & Admissible Heuristics")
    edges_df = pd.DataFrame(SearchService.get_graph_edges())
    heuristics_df = pd.DataFrame(list(search_res["heuristics"].items()), columns=["State Node", "Heuristic h(n) (Estimated Remaining Hours)"])

    g_col1, g_col2 = st.columns([6, 6])
    with g_col1:
        st.markdown("#### Transition Actions & Edge Costs")
        st.dataframe(edges_df, use_container_width=True, hide_index=True)
    with g_col2:
        st.markdown("#### Admissible Heuristic Table $h(n)$")
        st.dataframe(heuristics_df, use_container_width=True, hide_index=True)
