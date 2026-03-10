"""
astar.py — A* search algorithm (and Weighted A*).

"""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

from search.utils import Graph, SearchLogger, reconstruct_path


def astar(
    graph: Graph,
    start: Any,
    goal: Any,
    heuristic: Callable[[Any], float],
    weight: float = 1.0,
    allow_reopen: bool = True,
) -> Tuple[Optional[List], Optional[float], SearchLogger]:
    """
    A* (or Weighted A* when weight > 1).

    f(n) = g(n) + weight * h(n)

    Parameters
    ----------
    graph       : adjacency dict  node -> [(neighbour, cost)]
    start       : initial state
    goal        : goal state
    heuristic   : h(n) — should be admissible for optimality guarantee
    weight      : inflation factor (1 = standard A*)
    allow_reopen: if True, nodes are re-opened when a cheaper path is found
                  (necessary for non-consistent heuristics)

    Returns
    -------
    path, cost, logger
    """
    logger = SearchLogger(f"A* (w={weight})")
    logger.start()

    # g[n] = best cost found so far from start to n
    g: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Optional[Any]] = {start: None}
    closed: set = set()

    # heap entries: (f, tie-break counter, node)
    counter = 0
    h_start = heuristic(start)
    heap: List[Tuple[float, int, Any]] = [(h_start * weight, counter, start)]

    step = 0
    while heap:
        f_cur, _, node = heapq.heappop(heap)

        # Lazy deletion: skip stale entries
        if node in closed:
            # If allow_reopen, check if we have a better g
            if not allow_reopen or f_cur >= g.get(node, float("inf")) + weight * heuristic(node):
                continue

        closed.add(node)
        g_cur = g[node]
        h_cur = heuristic(node)

        logger.log_expansion(step, node, g_cur, h_cur, g_cur + weight * h_cur, len(heap))
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
                h_n = heuristic(neighbour)
                f_n = tentative_g + weight * h_n
                counter += 1
                heapq.heappush(heap, (f_n, counter, neighbour))
                # Allow re-opening
                if neighbour in closed and allow_reopen:
                    closed.discard(neighbour)

    logger.stop()
    return None, None, logger   # no path found
