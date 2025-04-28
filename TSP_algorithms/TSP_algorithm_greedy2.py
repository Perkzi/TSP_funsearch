"""
TSP greedy with full distance matrix in priority
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
    distances: np.ndarray
) -> float:
    """
    Calculate priority scores for all candidate cities at once.

    Args:
        start_idx: int: Index of the starting city (In the Traveling Salesman Problem (TSP),
            the solution requires the salesman to return to the starting city after visiting all other cities.).
        current_idx: int: Index of the current city.
        candidate_idx: int: Index of one candidate city (one selected city in unvisited cities).
        unvisited: np.ndarray: Indices of all unvisited cities
        distances: np.ndarray: Distances from any city to any other cities 
            (distances[candidate_idx][current_idx] represents the distance from candidate city to current city)
            (distances[current_idx][unvisited[0]] represents the distance from current city to an unvisited city).

    Returns:
        float: priority score for the candidate city.
                    Higher score means the city is more preferred.
    """
    direct_distance = distances[candidate_idx][current_idx]

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
            # distances_row = dist[candidate]
    
            # Compute priority scores for all unvisited cities
            priority = tsp_priority(
                start_idx=start,
                current_idx=tour[-1],
                candidate_idx=candidate,
                unvisited=unvisited,
                distances=dist
            )
            if priority>best_priority:
                best_candidate = candidate
                best_priority = priority 

        # Update tour
        tour.append(best_candidate)
        visited.add(best_candidate)
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

