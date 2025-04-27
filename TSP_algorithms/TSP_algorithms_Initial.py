specification = r'''
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
        cost += dist[route[i], route[(i+1)%n]]
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
    candidate_idx: int,
    distances_row: np.ndarray,
    candidate_idxs: List[int]
) -> float:
    """
    Calculate the priority score for a single candidate city from the current city.

    Args:
        start_idx (int): Index of the starting city (may be used for future enhancements).
        current_idx (int): Index of the current city (the city we are currently at).
        candidate_idx (int): Index of the candidate city to be evaluated.
        distances_row (np.ndarray): 1D array where distances_row[i] is the distance from current city to city i.
        candidate_idxs (List[int]): List of indices of all candidate (unvisited) cities.

    Returns:
        float: Priority score for the candidate city.
               Higher score means higher priority for selection.
    """
    # Distance from current city to candidate city
    distance = distances_row[candidate_idx]

    # We can also use all candidate distances if needed
    candidate_distances = distances_row[candidate_idxs]

    # Standardize based on candidate distances
    mean_d = np.mean(candidate_distances)
    std_d = np.std(candidate_distances) + 1e-9  # avoid division by zero

    z = distance

    # Priority design: cities with distance close to mean are prioritized
    priority = -np.log(np.abs(z) + 1e-9)

    return priority


# --------------------------- TSP solver --------------------------------- #

def tsp_solve(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    visited: set[int] = {start}
    tour: List[int] = [start]
    # mean_d, std_d = float(dist.mean()), float(dist.std())

    current = start
    for _ in range(n - 1):
        unvisited = _get_unvisited(n, visited)
        distances_row = dist[current]
        priorities = []
        for candidate_idx in unvisited:
            priority = tsp_priority(
                start_idx=start,
                current_idx=current,
                candidate_idx=candidate_idx,
                distances_row=distances_row,
                candidate_idxs=list(unvisited)
            )
            priorities.append(priority)
        
        next_city = int(unvisited[_arg_best(np.array(priorities))])
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
