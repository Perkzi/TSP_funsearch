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

def two_opt(route: list[int], distances: np.ndarray) -> list[int]:
    """
    Basic 2-opt optimizer: iteratively reverse segments of the route
    to eliminate crossings and shorten the total tour length,
    until no further improvement is possible.
    """
    n = len(route)
    best_route = route.copy()
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # Skip adjacent cities (no meaningful reversal)
                a, b = best_route[i - 1], best_route[i]
                c, d = best_route[j], best_route[(j + 1) % n]
                if distances[a, c] + distances[b, d] < distances[a, b] + distances[c, d]:
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    improved = True
        route = best_route.copy()

    return best_route

def tsp_priority(candidate_idx: int, current_tour: List[int], unvisited: np.ndarray, distances: np.ndarray) -> Tuple[float, int]:
    """
    Calculate priority scores for one candidate city.

    Args:
        candidate_idx: int: Index of one candidate city (one selected city in unvisited cities).
        current_tour: List[int]: Indies of current tour of TSP solving, start city index is current_tour[0],
            current end city index is current_tour[-1] (In the Traveling Salesman Problem (TSP),
            the solution requires the salesman to return to the start city after visiting all other cities.)
        unvisited: np.ndarray: Indices of all unvisited cities
        distances: np.ndarray: Distances from any city to any other cities 
            (distances[candidate_idx][current_tour[-1]] represents the distance from the candidate city to current end city)
            (distances[current_tour[-1]][unvisited[0]] represents the distance from current end city to an unvisited city).

    Returns:
        Tuple[float, int]:
            - float: The priority score for the candidate city (higher is more preferred).
                     This is computed as the negative of the minimal extra cost caused by insertion.
            - int: The index in current_tour where the candidate city should be inserted.
                     For example, an insertion index of 2 means the candidate is inserted between
                     current_tour[1] and current_tour[2].
                     If return None
    """
    """Even more improved version of tsp_priority function."""
    n = len(current_tour)
    
    if n < 2:
        cost_increase = 2 * distances[current_tour[0]][candidate_idx]
        best_insertion_idx = 1
    else:
        best_cost_increase = float('inf')
        best_insertion_idx = None
        
        for i in range(n):
            j = (i + 1) % n
            cost_increase = (distances[current_tour[i]][candidate_idx] +
                             distances[candidate_idx][current_tour[j]] -
                             distances[current_tour[i]][current_tour[j]])
            
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_insertion_idx = i + 1
    
    min_distance = float('inf')
    for city in current_tour:
        d = distances[candidate_idx][city]
        if d < min_distance:
            min_distance = d

    if n < 2:
        priority = min_distance
    else:
        priority = min_distance - (best_cost_increase / n)
    
    # Custom logic to penalize choosing closest city for next step
    if priority < 100:  # If the minimum distance is less than 100, penalize the choice
        priority *= 1.2
    
    return priority, best_insertion_idx

def tsp_solve(dist: np.ndarray, start: int = 0) -> list[int]:
    n = dist.shape[0]
    visited: set[int] = {start}
    tour: List[int] = [start]

    for _ in range(n - 1):
        # Find unvisited cities
        unvisited = _get_unvisited(n, visited)
        
        best_candidate = -1
        best_priority = -np.inf
        best_insertion_idx = len(tour)
            
        for candidate in unvisited:
            # Distances from candidate city to all cities
            # distances_row = dist[candidate]
    
            # Compute priority scores for all unvisited cities
            result = tsp_priority(
                candidate_idx=candidate,
                current_tour=tour,
                unvisited=unvisited,
                distances=dist
            )
            if isinstance(result, tuple):
                priority, insertion_idx = result
            else:
                priority = result
                insertion_idx = None
                
            if priority>best_priority:
                best_candidate = candidate
                best_priority = priority 
                if insertion_idx is not None:
                    best_insertion_idx = insertion_idx

        # Update tour
        tour.insert(best_insertion_idx, best_candidate)
        visited.add(best_candidate)
    tour = two_opt(tour, dist)  
    return tour # 不含重复首尾

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
            opt_len = float(distances[tour, tour[(idx + 1) % len(tour)]].sum())
    
        if opt_len is not None:
            approx = cost / opt_len
            print(f"    路径 = {cost:.0f}, 最优 = {opt_len:.0f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
        else:
            print(f"    路径 = {cost:.0f}, 时间 = {elapsed:.3f}s")

    return -np.mean(total_costs)
