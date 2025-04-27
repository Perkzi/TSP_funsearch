specification = r'''
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tsplib95

# --------------------------- Core TSP routines --------------------------- #
def tsp_evaluate(route: List[int], dist: np.ndarray) -> float:
    """Given a city tour and distance matrix, return total route length (including returning to start)."""
    route = np.asarray(route)
    n = len(route)
    cost = 0.0
    for i in range(n):
        cost += dist[route[i], route[(i+1) % n]]
    return float(cost)

# ---- small helpers ----------------------------------------------------- #

def _get_unvisited(n: int, visited: set[int]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[list(visited)] = False
    return np.nonzero(mask)[0]

def _arg_best(prio: np.ndarray) -> int:
    return int(np.argmax(prio))

# ----------------------- FunSearch evolve target ------------------------ #

# import funsearch  # type: ignore  # Provided by the FunSearch runtime

@funsearch.evolve
def tsp_priority(
    start_idx: int,
    current_idx: int,
    candidate_idxs: List[int],
    distances_row: np.ndarray
) -> np.ndarray:
    """
    Calculate priority scores for all candidate cities at once.

    Args:
        start_idx (int): Index of the starting city (reserved for future use).
        current_idx (int): Index of the current city.
        candidate_idxs (List[int]): List of indices of candidate cities (unvisited cities).
        distances_row (np.ndarray): Distances from the current city to all other cities.

    Returns:
        np.ndarray: An array of priority scores for the candidate cities.
                    Higher score means the city is more preferred.
    """
    candidate_distances = distances_row[candidate_idxs]

    # You can design any transformation here.
    mean_d = np.mean(candidate_distances)
    std_d = np.std(candidate_distances) + 1e-9

    z = (candidate_distances - mean_d) / std_d

    # Priority: cities closer to mean are preferred (you can customize here!)
    priorities = -np.log(np.abs(z) + 1e-9)

    return priorities

# --------------------------- TSP solver --------------------------------- #

def tsp_solve(dist: np.ndarray, start: int = 0) -> List[int]:
    """
    Solve TSP using greedy priority-based heuristic.

    Args:
        dist (np.ndarray): 2D array, distance matrix between cities. dist[i, j] is distance from city i to city j.
        start (int): Index of starting city. Default is 0.

    Returns:
        List[int]: The ordered list of city indices representing the tour.
    """
    n = dist.shape[0]
    visited: set[int] = {start}
    tour: List[int] = [start]

    current = start

    for _ in range(n - 1):
        # Find unvisited cities
        unvisited = _get_unvisited(n, visited)
        
        # Distances from current city to all cities
        distances_row = dist[current]

        # Compute priority scores for all unvisited cities
        priorities = tsp_priority(
            start_idx=start,
            current_idx=current,
            candidate_idxs=list(unvisited),
            distances_row=distances_row
        )

        # Select the unvisited city with the highest priority
        next_city_idx_in_unvisited = _arg_best(priorities)
        next_city = int(unvisited[next_city_idx_in_unvisited])

        # Update tour
        tour.append(next_city)
        visited.add(next_city)
        current = next_city

    return tour

# ------------------------ FunSearch evaluate ---------------------------- #

@funsearch.run  # FunSearch maximises this score
def evaluate(instance: Dict[str, object]) -> float:
    """评估单个实例，不是多个！"""
    print("Evaluating a single instance...")

    dist: np.ndarray = instance["distances"]  # type: ignore

    t0 = time.perf_counter()
    route = tsp_solve(dist)
    elapsed = time.perf_counter() - t0
    cost = tsp_evaluate(route, dist)

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

    return -cost  # 注意这里只返回单个cost，不再是mean了！
'''
