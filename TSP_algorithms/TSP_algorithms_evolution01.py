"""
TSP greedy with one distance row in priority
"""

specification = r'''
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple

def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    return (
        sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
        + distances[route[-1], route[0]]
    )
    
# ---- small helpers ----------------------------------------------------- #
def two_opt(route: list[int], distances: np.ndarray) -> list[int]:
    """
    简单版 2-opt 改进器：输入一条路径，局部翻转，直到无法再改进为止。
    """
    n = len(route)
    best_route = route.copy()
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:  # 连续的两个点，跳过，不反转
                    continue
                # 原先的两条边是 (i-1 → i) 和 (j → j+1)
                # 反转后是 (i-1 → j) 和 (i → j+1)
                a, b = best_route[i - 1], best_route[i]
                c, d = best_route[j], best_route[(j + 1) % n]
                if distances[a, c] + distances[b, d] < distances[a, b] + distances[c, d]:
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    improved = True
        route = best_route.copy()

    return best_route
def _get_unvisited(n: int, visited: set[int]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[list(visited)] = False
    return np.nonzero(mask)[0]

def _arg_best(prio: np.ndarray) -> int:
    return int(np.argmax(prio))
    
@funsearch.evolve
def tsp_priority(
    start_idx: int,
    current_idx: int,
    candidate_idx: int,
    unvisited: np.ndarray,
    distances_row: np.ndarray
) -> float:
    """
    Calculate priority scores for all candidate cities at once.

    Args:
        start_idx: int: Index of the starting city (In the Traveling Salesman Problem (TSP),
            the solution requires the salesman to return to the starting city after visiting all other cities.).
        current_idx: int: Index of the current city.
        candidate_idx: int: Index of one candidate city (one selected city in unvisited cities).
        unvisited: np.ndarray: Indices of all unvisited cities
        distances_row: np.ndarray: Distances from the candidate city to all other cities 
            (distances_row[current_idx] represents the distance from candidate city to current city)
            (distances_row[unvisited[0]] represents the distance from candidate city to another unvisited city).

    Returns:
        float: priority score for the candidate city.
                    Higher score means the city is more preferred.
    """
    direct_distance = distances_row[current_idx]

    return -direct_distance

def tsp_solve(dist: np.ndarray, start: int = 0) -> list[int]:
    n = dist.shape[0]
    visited: set[int] = {start}
    tour: List[int] = [start]

    for _ in range(n - 1):
        # Find unvisited cities
        unvisited = _get_unvisited(n, visited)
        
        best_candidate = -1
        best_priority = -np.inf
            
        for candidate in unvisited:
            # Distances from candidate city to all cities
            distances_row = dist[candidate]
    
            # Compute priority scores for all unvisited cities
            priority = tsp_priority(
                start_idx=start,
                current_idx=tour[-1],
                candidate_idx=candidate,
                unvisited=unvisited,
                distances_row=distances_row
            )
            if priority>best_priority:
                best_candidate = candidate
                best_priority = priority 

        # Update tour
        tour.append(best_candidate)
        visited.add(best_candidate)
tour = two_opt(tour, dist) # 加2-opt微调
return tour # 不含重复首尾

    return tour # 不含重复首尾

@funsearch.run
def evaluate(instances: dict) -> float:
    total_costs, total_times = [], []
    compare_with_optimal = True
    summary = []

    for name, instance in instances.items():
        distances = instance["distances"]

        start_time = time.perf_counter()
        route = tsp_solve(distances)
        elapsed = time.perf_counter() - start_time

        cost = tsp_evaluate(route, distances)
        total_costs.append(cost)
        total_times.append(elapsed)
        
        # Optimal reference (optional)
        opt_len: float | None = None
        if "optimal_tour" in instance and instance["optimal_tour"] is not None:
            tour = np.asarray(instance["optimal_tour"]).flatten()
            idx = np.arange(len(tour))
            opt_len = float(dist[tour, tour[(idx + 1) % len(tour)]].sum())
    
        if opt_len is not None:
            approx = cost / opt_len
            print(f"    路径 = {cost:.0f}, 最优 = {opt_len:.0f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
        else:
            print(f"    路径 = {cost:.0f}, 时间 = {elapsed:.3f}s")

    return -np.mean(total_costs)
'''
