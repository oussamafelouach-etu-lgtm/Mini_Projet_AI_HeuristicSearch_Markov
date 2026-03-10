"""
greedy.py — Greedy Best-First Search.

f(n) = h(n)   (ignores actual path cost g)

NOT guaranteed to find the optimal path.
"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from search.utils import Graph, SearchLogger, reconstruct_path


def greedy_best_first(
    graph: Graph,
    start: Any,
    goal: Any,
    heuristic: Callable[[Any], float],
) -> Tuple[Optional[List], Optional[float], SearchLogger]:
    """
    Greedy Best-First Search.

    Expands the node that appears closest to the goal according to h.
    Fast in practice but may return sub-optimal paths.

    Returns
    -------
    path, cost (actual g along returned path), logger
    """
    logger = SearchLogger("Greedy Best-First")
    logger.start()

    g: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Optional[Any]] = {start: None}
    closed: set = set()

    counter = 0
    h_start = heuristic(start)
    heap: List[Tuple[float, int, Any]] = [(h_start, counter, start)]

    step = 0
    while heap:
        h_cur_val, _, node = heapq.heappop(heap)

        if node in closed:
            continue
        closed.add(node)

        g_cur = g[node]
        h_cur = heuristic(node)
        logger.log_expansion(step, node, g_cur, h_cur, h_cur, len(heap))
        step += 1

        if node == goal:
            path = reconstruct_path(came_from, goal)
            logger.stop()
            return path, g_cur, logger

        for neighbour, cost in graph.get(node, []):
            if neighbour not in closed:
                tentative_g = g_cur + cost
                # Only update g for path reconstruction accuracy
                if tentative_g < g.get(neighbour, float("inf")):
                    g[neighbour] = tentative_g
                    came_from[neighbour] = node
                h_n = heuristic(neighbour)
                counter += 1
                heapq.heappush(heap, (h_n, counter, neighbour))

    logger.stop()
    return None, None, logger
