"""Search Algorithms Module implementing BFS, DFS, and A* on a structured state space."""
import heapq
import time
from collections import deque
from typing import List, Dict, Any, Tuple, Optional, Set

# Enterprise Domain State-Space Problem:
# Customer Feedback Escalation and Resolution Workflow Graph
# Nodes represent discrete resolution operational stages; edges represent transition actions with costs (labor hours).

WORKFLOW_GRAPH: Dict[str, List[Tuple[str, float]]] = {
    "Complaint Received": [
        ("Automated Triage", 1.0),
        ("Customer Care Hotline", 2.0)
    ],
    "Automated Triage": [
        ("Quality Investigation", 3.0),
        ("Logistics Carrier Audit", 2.0),
        ("Customer Service Intervention", 1.5)
    ],
    "Customer Care Hotline": [
        ("Customer Service Intervention", 2.0),
        ("Direct Refund Processing", 1.0)
    ],
    "Quality Investigation": [
        ("Engineering Root Cause Analysis", 5.0),
        ("Batch Sample Recall", 4.0)
    ],
    "Logistics Carrier Audit": [
        ("Warehouse Packaging Redesign", 3.0),
        ("Carrier SLA Penalty Review", 2.0)
    ],
    "Customer Service Intervention": [
        ("Executive Customer Outreach", 2.5),
        ("Resolution Verified & Closed", 4.0)
    ],
    "Direct Refund Processing": [
        ("Resolution Verified & Closed", 3.0)
    ],
    "Engineering Root Cause Analysis": [
        ("Firmware/Component Patch", 4.0),
        ("Executive Review & Policy Update", 3.0)
    ],
    "Batch Sample Recall": [
        ("Executive Review & Policy Update", 2.0)
    ],
    "Warehouse Packaging Redesign": [
        ("Executive Review & Policy Update", 2.0)
    ],
    "Carrier SLA Penalty Review": [
        ("Resolution Verified & Closed", 2.5)
    ],
    "Executive Customer Outreach": [
        ("Resolution Verified & Closed", 1.5)
    ],
    "Firmware/Component Patch": [
        ("Resolution Verified & Closed", 2.0)
    ],
    "Executive Review & Policy Update": [
        ("Resolution Verified & Closed", 1.0)
    ],
    "Resolution Verified & Closed": []
}

# Admissible heuristic values h(n) estimating remaining effort (hours) to goal 'Resolution Verified & Closed'
WORKFLOW_HEURISTICS: Dict[str, float] = {
    "Complaint Received": 8.0,
    "Automated Triage": 6.5,
    "Customer Care Hotline": 6.0,
    "Quality Investigation": 6.0,
    "Logistics Carrier Audit": 4.5,
    "Customer Service Intervention": 4.0,
    "Direct Refund Processing": 3.0,
    "Engineering Root Cause Analysis": 4.0,
    "Batch Sample Recall": 3.0,
    "Warehouse Packaging Redesign": 3.0,
    "Carrier SLA Penalty Review": 2.5,
    "Executive Customer Outreach": 1.5,
    "Firmware/Component Patch": 2.0,
    "Executive Review & Policy Update": 1.0,
    "Resolution Verified & Closed": 0.0
}

def run_bfs(start_state: str, goal_state: str) -> Dict[str, Any]:
    """Breadth-First Search (FIFO Queue) exploring shallowest paths first."""
    start_time = time.perf_counter()
    queue = deque([[start_state]])
    visited: Set[str] = {start_state}
    expanded_nodes: List[str] = []

    while queue:
        path = queue.popleft()
        node = path[-1]
        expanded_nodes.append(node)

        if node == goal_state:
            exec_time = time.perf_counter() - start_time
            # Calculate path cost
            cost = _compute_path_cost(path)
            return {
                "algorithm": "Breadth-First Search (BFS)",
                "success": True,
                "path": path,
                "path_cost": cost,
                "expanded_nodes_count": len(expanded_nodes),
                "expanded_nodes": expanded_nodes,
                "execution_time_ms": round(exec_time * 1000, 3),
                "goal_found": True
            }

        for neighbor, _ in WORKFLOW_GRAPH.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    exec_time = time.perf_counter() - start_time
    return {
        "algorithm": "Breadth-First Search (BFS)",
        "success": False,
        "path": [],
        "path_cost": 0.0,
        "expanded_nodes_count": len(expanded_nodes),
        "expanded_nodes": expanded_nodes,
        "execution_time_ms": round(exec_time * 1000, 3),
        "goal_found": False
    }

def run_dfs(start_state: str, goal_state: str) -> Dict[str, Any]:
    """Depth-First Search (LIFO Stack) exploring deep paths first."""
    start_time = time.perf_counter()
    stack = [[start_state]]
    visited: Set[str] = set()
    expanded_nodes: List[str] = []

    while stack:
        path = stack.pop()
        node = path[-1]

        if node not in visited:
            visited.add(node)
            expanded_nodes.append(node)

            if node == goal_state:
                exec_time = time.perf_counter() - start_time
                cost = _compute_path_cost(path)
                return {
                    "algorithm": "Depth-First Search (DFS)",
                    "success": True,
                    "path": path,
                    "path_cost": cost,
                    "expanded_nodes_count": len(expanded_nodes),
                    "expanded_nodes": expanded_nodes,
                    "execution_time_ms": round(exec_time * 1000, 3),
                    "goal_found": True
                }

            # Add neighbors in reverse order for standard traversal
            neighbors = WORKFLOW_GRAPH.get(node, [])
            for neighbor, _ in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(path + [neighbor])

    exec_time = time.perf_counter() - start_time
    return {
        "algorithm": "Depth-First Search (DFS)",
        "success": False,
        "path": [],
        "path_cost": 0.0,
        "expanded_nodes_count": len(expanded_nodes),
        "expanded_nodes": expanded_nodes,
        "execution_time_ms": round(exec_time * 1000, 3),
        "goal_found": False
    }

def run_astar(start_state: str, goal_state: str) -> Dict[str, Any]:
    """A* Informed Search evaluating f(n) = g(n) + h(n)."""
    start_time = time.perf_counter()
    # Priority queue stores tuples: (f_score, g_score, path)
    initial_h = WORKFLOW_HEURISTICS.get(start_state, 0.0)
    pq = [(initial_h, 0.0, [start_state])]
    g_costs: Dict[str, float] = {start_state: 0.0}
    expanded_nodes: List[str] = []

    while pq:
        f, g, path = heapq.heappop(pq)
        node = path[-1]
        expanded_nodes.append(node)

        if node == goal_state:
            exec_time = time.perf_counter() - start_time
            return {
                "algorithm": "A* Informed Search",
                "success": True,
                "path": path,
                "path_cost": g,
                "expanded_nodes_count": len(expanded_nodes),
                "expanded_nodes": expanded_nodes,
                "execution_time_ms": round(exec_time * 1000, 3),
                "goal_found": True
            }

        for neighbor, edge_cost in WORKFLOW_GRAPH.get(node, []):
            new_g = g + edge_cost
            if neighbor not in g_costs or new_g < g_costs[neighbor]:
                g_costs[neighbor] = new_g
                h = WORKFLOW_HEURISTICS.get(neighbor, 0.0)
                new_f = new_g + h
                heapq.heappush(pq, (new_f, new_g, path + [neighbor]))

    exec_time = time.perf_counter() - start_time
    return {
        "algorithm": "A* Informed Search",
        "success": False,
        "path": [],
        "path_cost": 0.0,
        "expanded_nodes_count": len(expanded_nodes),
        "expanded_nodes": expanded_nodes,
        "execution_time_ms": round(exec_time * 1000, 3),
        "goal_found": False
    }

def _compute_path_cost(path: List[str]) -> float:
    cost = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        for neighbor, edge_cost in WORKFLOW_GRAPH.get(u, []):
            if neighbor == v:
                cost += edge_cost
                break
    return round(cost, 2)
