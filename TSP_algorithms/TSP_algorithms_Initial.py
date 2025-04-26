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
def tsp_priority(distances_row: np.ndarray, mean_d: float, std_d: float) -> np.ndarray:
    """给定当前城市到候选城市距离向量，返回优先度（越大越好）。

    这里用 -((d-mean)/std)^2 公式，FunSearch 会自动进化更好的表达式。"""
    z = (distances_row - mean_d) / (std_d + 1e-9)
    # print(z)
    return -z ** 2

# --------------------------- TSP solver --------------------------------- #

def tsp_solve(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    visited: set[int] = {start}
    tour: List[int] = [start]
    mean_d, std_d = float(dist.mean()), float(dist.std())

    current = start
    for _ in range(n - 1):
        unvisited = _get_unvisited(n, visited)
        prio = tsp_priority(dist[current, unvisited], mean_d, std_d)
        next_city = int(unvisited[_arg_best(prio)])
        tour.append(next_city)
        visited.add(next_city)
        current = next_city
    return tour

# ------------------------ FunSearch evaluate ---------------------------- #

@funsearch.run  # FunSearch maximises this score
def evaluate(instances: Dict[str, Dict[str, object]]) -> float:
    total_costs: List[float] = []
    total_times: List[float] = []

    for name, inst in instances.items():
        dist: np.ndarray = inst["distances"]  # type: ignore

        t0 = time.perf_counter()
        route = tsp_solve(dist)
        elapsed = time.perf_counter() - t0
        cost = tsp_evaluate(route, dist)

        # Optimal reference (optional)
        opt_len: float | None = None
        if "optimal_tour" in inst and inst["optimal_tour"] is not None:
            tour = np.asarray(inst["optimal_tour"]).flatten()
            # tour = inst["optimal_tour"]
            idx = np.arange(len(tour))
            opt_len = float(dist[tour, tour[(idx + 1) % len(tour)]].sum())

        if opt_len is not None:
            approx = cost / opt_len
            print(f"{name}: 路径 = {cost:.0f}, 最优 = {opt_len:.0f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
        else:
            print(f"{name}: 路径 = {cost:.0f}, 时间 = {elapsed:.3f}s")

        total_costs.append(cost)
        total_times.append(elapsed)

    # ----- 计算平均得分（负号 → 最小化路径长度）----- #
    if not total_costs:
        return -1e9  # 严罚空实例

    mean_cost = float(np.nanmean(total_costs))
    if not np.isfinite(mean_cost):
        return -1e9

    print(f"平均路径 = {mean_cost:.1f}, 平均时间 = {np.mean(total_times):.3f}s")
    return -mean_cost
'''
