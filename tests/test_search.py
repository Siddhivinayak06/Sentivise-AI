"""Unit tests for Search Algorithms Lab: BFS, DFS, and A* search implementations."""
import pytest
from modules.search_algorithms import run_bfs, run_dfs, run_astar
from services.search_service import SearchService

def test_search_algorithms_reach_goal():
    start = "Complaint Received"
    goal = "Resolution Verified & Closed"

    bfs = run_bfs(start, goal)
    assert bfs["success"]
    assert bfs["path"][0] == start
    assert bfs["path"][-1] == goal
    assert bfs["path_cost"] > 0

    dfs = run_dfs(start, goal)
    assert dfs["success"]
    assert dfs["path"][0] == start
    assert dfs["path"][-1] == goal

    astar = run_astar(start, goal)
    assert astar["success"]
    assert astar["path"][0] == start
    assert astar["path"][-1] == goal
    assert astar["path_cost"] <= bfs["path_cost"]  # A* cost is optimal

def test_search_service_comparison_table():
    comp = SearchService.compare_search_algorithms("Complaint Received", "Resolution Verified & Closed")
    assert "bfs" in comp
    assert "dfs" in comp
    assert "astar" in comp
    assert len(comp["comparison"]) == 3
