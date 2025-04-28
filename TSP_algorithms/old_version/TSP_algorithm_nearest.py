specification = r'''
import os
import gzip
import numpy as np
import tsplib95
import time
import pandas as pd

def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    return sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1)) + distances[route[-1], route[0]]

def tsp_solve(distances: np.ndarray, start_city: int = 0) -> list[int]:
    num_cities = distances.shape[0]
    visited = [start_city]
    current_city = start_city
    for _ in range(num_cities - 1):
        unvisited = [i for i in range(num_cities) if i not in visited]
        next_city = tsp_priority(current_city, unvisited, distances)
        visited.append(next_city)
        current_city = next_city
    return visited

@funsearch.evolve
def tsp_priority(current_city: int, unvisited: list[int], distances: np.ndarray) -> int:
    best_cost = float("inf")
    best_city = unvisited[0]
    for city in unvisited:
        cost = distances[current_city, city]
        if cost < best_cost:
            best_cost = cost
            best_city = city
    return best_city

@funsearch.run
def evaluate(instances: dict) -> float:
    total_costs = []
    total_times = []
    compare_with_optimal = True  # 可选：如果不需要对比最优路径，请将此行改为 False 或注释掉
    summary = []

    for name, instance in instances.items():
        distances = instance["distances"]
        start = time.perf_counter()
        route = tsp_solve(distances)
        elapsed = time.perf_counter() - start

        cost = tsp_evaluate(route, distances)
        total_costs.append(cost)
        total_times.append(elapsed)

        opt_cost = instance.get("optimal_tour")
        if compare_with_optimal and opt_cost:
            opt_cost_val = tsp_evaluate(opt_cost, distances)
            approx = cost / opt_cost_val
            print(f"{name}: 路径长度 = {cost:.2f}, 最优 = {opt_cost_val:.2f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, opt_cost_val, approx, elapsed))
        else:
            print(f"{name}: 路径长度 = {cost:.2f}, 无最优参考, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, None, None, elapsed))

    print(f"平均路径长度: {np.mean(total_costs):.2f}, 平均运行时间: {np.mean(total_times):.3f}s")
    return -np.mean(total_costs)
'''


'''
import os
import numpy as np
import tsplib95
import funsearch
import time
import pandas as pd
def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    return sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1)) + distances[route[-1], route[0]]

def tsp_solve(distances: np.ndarray, start_city: int = 0) -> list[int]:
    num_cities = distances.shape[0]
    visited = [start_city]
    current_city = start_city
    for _ in range(num_cities - 1):
        unvisited = [i for i in range(num_cities) if i not in visited]
        next_city = tsp_priority(current_city, unvisited, distances)
        visited.append(next_city)
        current_city = next_city
    return visited

@funsearch.evolve
def tsp_priority(current_city: int, unvisited: list[int], distances: np.ndarray) -> int:
    best_cost = float("inf")
    best_city = unvisited[0]
    for city in unvisited:
        cost = distances[current_city, city]
        if cost < best_cost:
            best_cost = cost
            best_city = city
    return best_city

@funsearch.run
def evaluate(instances: dict) -> float:
    total_costs = []
    total_times = []
    compare_with_optimal = True  # 可选：如果不需要对比最优路径，请将此行改为 False 或注释掉
    summary = []

    for name, instance in instances.items():
        distances = instance["distances"]
        start = time.perf_counter()
        route = tsp_solve(distances)
        elapsed = time.perf_counter() - start

        cost = tsp_evaluate(route, distances)
        total_costs.append(cost)
        total_times.append(elapsed)

        opt_cost = instance.get("optimal_tour")
        if compare_with_optimal and opt_cost:
            opt_cost_val = tsp_evaluate(opt_cost, distances)
            approx = cost / opt_cost_val
            print(f"{name}: 路径长度 = {cost:.2f}, 最优 = {opt_cost_val:.2f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, opt_cost_val, approx, elapsed))
        else:
            print(f"{name}: 路径长度 = {cost:.2f}, 无最优参考, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, None, None, elapsed))

    print(f"平均路径长度: {np.mean(total_costs):.2f}, 平均运行时间: {np.mean(total_times):.3f}s")
    # df = pd.DataFrame(summary, columns=["Instance", "Route Cost", "Optimal Cost", "Approx Ratio", "Time (s)"])
    # df.to_csv("evaluation_summary.csv", index=False)
    return -np.mean(total_costs)

# 用法：一定先加载数据集
# evaluate(instances_AlgorithmDevelop)
# evaluate(instances_PerformanceTesting)
# evaluate(instances_GeneralizationTesting)
'''