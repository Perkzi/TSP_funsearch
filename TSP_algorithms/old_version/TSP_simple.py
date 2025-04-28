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
def tsp_priority(distance: float, mean_d: float, std_d: float) -> float:
    """
    Calculates the priority score for a candidate city based on its distance.

    Args:
        distance (float): The distance from the current city to the candidate city.
        mean_d (float): The mean distance between cities (not used in this version, but kept for compatibility).
        std_d (float): The standard deviation of distances between cities (not used in this version, but kept for compatibility).

    Returns:
        float: A priority score for the candidate city.
               Higher priority (less negative) indicates the city is more preferable to visit next.

    Notes:
        - Priority is calculated as the negative logarithm of the distance, squared.
        - Closer cities (smaller distance) will have higher priority scores.
        - A small constant (1e-9) is added inside the log to avoid numerical issues when distance is very close to zero.
    """
    z = distance
    return -np.log(np.abs(z) + 1e-9)** 2

# --------------------------- TSP solver --------------------------------- #
def tsp_solve(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    visited: set[int] = {start}
    tour: List[int] = [start]
    mean_d, std_d = float(dist.mean()), float(dist.std())

    current = start
    for _ in range(n - 1):
        unvisited = _get_unvisited(n, visited)
        priorities = np.array([
            tsp_priority(dist[current, city], mean_d, std_d) for city in unvisited
        ])
        next_city = int(unvisited[_arg_best(priorities)])
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
