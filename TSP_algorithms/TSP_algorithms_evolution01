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
    # --- 1. 基础部分：比较当前城市到候选城市的直线距离 和 未访问城市的平均距离 ---
    n_cities = len(unvisited)
    total_distance = 0.0
    for i in range(n_cities):
        total_distance += distances_row[unvisited[i]]

    avg_distance = total_distance / n_cities
    direct_distance = distances_row[candidate_idx]

    if direct_distance < avg_distance:
        priority_score = -(direct_distance - avg_distance)
    else:
        priority_score = -(direct_distance + avg_distance)

    # --- 2. 特殊条件：优先起点相关或者特定距离较小的城市 ---
    if start_idx == current_idx:
        priority_score += 10
    elif start_idx == candidate_idx:
        priority_score += 5
    elif direct_distance < avg_distance / 2:
        priority_score += 3

    # --- 3. 路径优化逻辑：倾向于离当前更近的城市 ---
    path_optimization_score = 0
    for i in range(n_cities):
        if distances_row[unvisited[i]] < avg_distance:
            path_optimization_score += 1
    priority_score += path_optimization_score

    # --- 4. 环路行为优化：优先靠近起点的城市 ---
    if candidate_idx == start_idx:
        priority_score += 7
    elif abs(candidate_idx - start_idx) == 1 or abs(candidate_idx - start_idx) == n_cities - 1:
        priority_score += 4

    return priority_score

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
