"""
ucs.py — Uniform-Cost Search (Dijkstra-style).

f(n) = g(n)   (heuristic h ≡ 0)
"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from search.utils import Graph, SearchLogger, reconstruct_path


def ucs(
    graph: Graph,
    start: Any,
    goal: Any,
) -> Tuple[Optional[List], Optional[float], SearchLogger]:
    """
    Uniform-Cost Search.

    Expands the node with the lowest cumulative path cost.
    Guaranteed to find the optimal path.

    Returns
    -------
    path, cost, logger
    """
    logger = SearchLogger("UCS")
    logger.start()

    g: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Optional[Any]] = {start: None}
    closed: set = set()

    counter = 0
    heap: List[Tuple[float, int, Any]] = [(0.0, counter, start)]

    step = 0
    while heap:
        g_cur, _, node = heapq.heappop(heap)

        if node in closed:
            continue
        closed.add(node)

        logger.log_expansion(step, node, g_cur, 0.0, g_cur, len(heap))
        step += 1

        if node == goal:
            path = reconstruct_path(came_from, goal)
            logger.stop()
            return path, g_cur, logger

        for neighbour, cost in graph.get(node, []):
            tentative_g = g_cur + cost
            if tentative_g < g.get(neighbour, float("inf")):
                g[neighbour] = tentative_g
                came_from[neighbour] = node
                counter += 1
                heapq.heappush(heap, (tentative_g, counter, neighbour))

    logger.stop()
    return None, None, logger
