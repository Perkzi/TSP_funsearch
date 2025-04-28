

specification = r'''
import os
import numpy as np
import tsplib95

import time
import pandas as pd

def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    return sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1)) + distances[route[-1], route[0]]

def tsp_solve(distances: np.ndarray, start_city: int = 0) -> list[int]:
    num_cities = distances.shape[0]
    # 初始回路：任选两个城市
    current_path = [start_city, (start_city + 1) % num_cities]
    unvisited = [c for c in range(num_cities) if c not in current_path]

    while unvisited:
        city_to_insert, pos = tsp_priority(current_path, unvisited, distances)
        current_path.insert(pos, city_to_insert)
        unvisited.remove(city_to_insert)

    return current_path   # 不含重复首尾；tsp_evaluate 会自动闭环


@funsearch.evolve
def tsp_priority(current_path: list[int],
                 unvisited: list[int],
                 distances: np.ndarray) -> tuple[int, int]:
    min_d_to_path = {}
    for city in unvisited:
        min_d_to_path[city] = min(distances[city, v] for v in current_path)

    city_far = max(min_d_to_path, key=min_d_to_path.get)  # 离回路最远的点
    best_delta = float("inf")
    best_pos = 1
    path_len = len(current_path)
    for idx in range(path_len):
        i = current_path[idx]
        j = current_path[(idx + 1) % path_len]          # 闭环
        delta = distances[i, city_far] + distances[city_far, j] - distances[i, j]
        if delta < best_delta:
            best_delta = delta
            best_pos = idx + 1

    return city_far, best_pos

@funsearch.run
def evaluate(instances: dict) -> float:
    total_costs = []
    total_times = []
    compare_with_optimal = True  # Toggle this to disable optimal comparison
    summary = []

    for name, instance in instances.items():
        distances = instance["distances"]
        start = time.perf_counter()
        route = tsp_solve(distances)
        elapsed = time.perf_counter() - start

        cost = tsp_evaluate(route, distances)
        total_costs.append(cost)
        total_times.append(elapsed)

        if compare_with_optimal and instance.get("optimal_tour"):
            opt_cost = tsp_evaluate(instance["optimal_tour"], distances)
            approx = cost / opt_cost
            print(f"{name}: 路径长度 = {cost:.2f}, 最优路径 = {opt_cost:.2f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, opt_cost, approx, elapsed))
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

import time
import pandas as pd
def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    return sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1)) + distances[route[-1], route[0]]

def tsp_solve(distances: np.ndarray, start_city: int = 0) -> list[int]:
    num_cities = distances.shape[0]
    # 初始回路：任选两个城市
    current_path = [start_city, (start_city + 1) % num_cities]
    unvisited = [c for c in range(num_cities) if c not in current_path]

    while unvisited:
        city_to_insert, pos = tsp_priority(current_path, unvisited, distances)
        current_path.insert(pos, city_to_insert)
        unvisited.remove(city_to_insert)

    return current_path   # 不含重复首尾；tsp_evaluate 会自动闭环


@funsearch.evolve
def tsp_priority(current_path: list[int],
                 unvisited: list[int],
                 distances: np.ndarray) -> tuple[int, int]:
    """
    经典 Farthest Insertion：
    1) 选 city_far —— 它到“已在回路中的城市集合”的最近距离最大。
    2) 在回路中找一条边 (i -> j)，把 city_far 插入 i-city-j，使
       增量 d(i,city)+d(city,j)-d(i,j) 最小。
    返回 (city_far, insert_position)：
        insert_position = 把 city_far 插到 current_path 的哪个索引之前。
    """
    # ---------- 第 1 步：挑选最远城市 ----------
    # 计算每个未访问城市到回路中所有城市的最小距离
    min_d_to_path = {}
    for city in unvisited:
        min_d_to_path[city] = min(distances[city, v] for v in current_path)

    city_far = max(min_d_to_path, key=min_d_to_path.get)  # 离回路最远的点

    # ---------- 第 2 步：在回路中找最佳插入位置 ----------
    best_delta = float("inf")
    best_pos = 1
    path_len = len(current_path)
    for idx in range(path_len):
        i = current_path[idx]
        j = current_path[(idx + 1) % path_len]          # 闭环
        delta = distances[i, city_far] + distances[city_far, j] - distances[i, j]
        if delta < best_delta:
            best_delta = delta
            best_pos = idx + 1

    return city_far, best_pos

@funsearch.run
def evaluate(instances: dict) -> float:
    total_costs = []
    total_times = []
    compare_with_optimal = True  # Toggle this to disable optimal comparison
    summary = []

    for name, instance in instances.items():
        distances = instance["distances"]
        start = time.perf_counter()
        route = tsp_solve(distances)
        elapsed = time.perf_counter() - start

        cost = tsp_evaluate(route, distances)
        total_costs.append(cost)
        total_times.append(elapsed)

        if compare_with_optimal and instance.get("optimal_tour"):
            opt_cost = tsp_evaluate(instance["optimal_tour"], distances)
            approx = cost / opt_cost
            print(f"{name}: 路径长度 = {cost:.2f}, 最优路径 = {opt_cost:.2f}, 近似比 = {approx:.4f}, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, opt_cost, approx, elapsed))
        else:
            print(f"{name}: 路径长度 = {cost:.2f}, 无最优参考, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, None, None, elapsed))

    print(f"平均路径长度: {np.mean(total_costs):.2f}, 平均运行时间: {np.mean(total_times):.3f}s")
    #df = pd.DataFrame(summary, columns=["Instance", "Route Cost", "Optimal Cost", "Approx Ratio", "Time (s)"])
    #df.to_csv("evaluation_summary.csv", index=False)
    return -np.mean(total_costs)

# 用法：
# evaluate(instances_AlgorithmDevelop)
# evaluate(instances_PerformanceTesting)
# evaluate(instances_GeneralizationTesting)
'''