"""Search Lab Service managing graph exploration and comparative benchmarking."""
from typing import Dict, Any, List
from modules.search_algorithms import (
    run_bfs, run_dfs, run_astar, WORKFLOW_GRAPH, WORKFLOW_HEURISTICS
)

class SearchService:
    @staticmethod
    def get_available_states() -> List[str]:
        return list(WORKFLOW_GRAPH.keys())

    @staticmethod
    def get_graph_edges() -> List[Dict[str, Any]]:
        edges = []
        for source, targets in WORKFLOW_GRAPH.items():
            for dest, cost in targets:
                edges.append({
                    "source": source,
                    "target": dest,
                    "cost": cost
                })
        return edges

    @staticmethod
    def compare_search_algorithms(start_state: str, goal_state: str) -> Dict[str, Any]:
        """Runs BFS, DFS, and A* concurrently on the selected states and compares performance."""
        bfs_res = run_bfs(start_state, goal_state)
        dfs_res = run_dfs(start_state, goal_state)
        astar_res = run_astar(start_state, goal_state)

        comparison_table = [
            {
                "Algorithm": "Breadth-First Search (BFS)",
                "Status": "Success" if bfs_res["success"] else "Failed",
                "Path Length (Steps)": len(bfs_res["path"]),
                "Path Cost (Hours)": bfs_res["path_cost"],
                "Expanded Nodes": bfs_res["expanded_nodes_count"],
                "Runtime (ms)": bfs_res["execution_time_ms"],
                "Optimality Guarantee": "Shortest Step Count (Unweighted)"
            },
            {
                "Algorithm": "Depth-First Search (DFS)",
                "Status": "Success" if dfs_res["success"] else "Failed",
                "Path Length (Steps)": len(dfs_res["path"]),
                "Path Cost (Hours)": dfs_res["path_cost"],
                "Expanded Nodes": dfs_res["expanded_nodes_count"],
                "Runtime (ms)": dfs_res["execution_time_ms"],
                "Optimality Guarantee": "Non-Optimal"
            },
            {
                "Algorithm": "A* Informed Search",
                "Status": "Success" if astar_res["success"] else "Failed",
                "Path Length (Steps)": len(astar_res["path"]),
                "Path Cost (Hours)": astar_res["path_cost"],
                "Expanded Nodes": astar_res["expanded_nodes_count"],
                "Runtime (ms)": astar_res["execution_time_ms"],
                "Optimality Guarantee": "Optimal Cost (Admissible Heuristic)"
            }
        ]

        return {
            "bfs": bfs_res,
            "dfs": dfs_res,
            "astar": astar_res,
            "comparison": comparison_table,
            "heuristics": WORKFLOW_HEURISTICS
        }
