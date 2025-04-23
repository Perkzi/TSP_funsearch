import numpy as np
import time
import funsearch
specification = r'''
import numpy as np
import time
import funsearch

def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    return (
        sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
        + distances[route[-1], route[0]]
    )

@funsearch.evolve
def tsp_priority(
    current_path: list[int],          # 目前已形成的回路（首尾不同）
    unvisited: list[int],             # 仍待插入的城市
    distances: np.ndarray,            # 距离矩阵
) -> tuple[int, int]:
    """
    选择 (city_to_insert, insert_position)，使新增距离增量最小。
    insert_position 表示要把 city 插到 current_path 的哪个索引之前。
    """
    best_delta = float("inf")
    best_city, best_position = unvisited[0], 1

    path_len = len(current_path)
    for candidate_city in unvisited:
        # 枚举 current_path 上的每条边 (i -> j)
        for idx in range(path_len):
            i = current_path[idx]
            j = current_path[(idx + 1) % path_len]  # % 保证最后连回起点
            delta = (
                distances[i, candidate_city]
                + distances[candidate_city, j]
                - distances[i, j]
            )
            if delta < best_delta:
                best_delta = delta
                best_city = candidate_city
                best_position = idx + 1          # 插到 j 前面

    return best_city, best_position

def tsp_solve(distances: np.ndarray, start_city: int = 0) -> list[int]:
   
    num_cities = distances.shape[0]

    # --- ① 初始回路：任选两个城市 ---
    current_path = [start_city, (start_city + 1) % num_cities]
    unvisited = [c for c in range(num_cities) if c not in current_path]

    # --- ② 逐步插入剩余城市 ---
    while unvisited:
        city_to_insert, insert_pos = tsp_priority(current_path, unvisited, distances)
        current_path.insert(insert_pos, city_to_insert)
        unvisited.remove(city_to_insert)

    return current_path  # 不含重复首尾

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

        opt_route = instance.get("optimal_tour")
        if compare_with_optimal and opt_route:
            opt_cost = tsp_evaluate(opt_route, distances)
            approx_ratio = cost / opt_cost
            print(
                f"{name}: 路径长度 = {cost:.2f}, "
                f"最优 = {opt_cost:.2f}, 近似比 = {approx_ratio:.4f}, "
                f"时间 = {elapsed:.3f}s"
            )
            summary.append((name, cost, opt_cost, approx_ratio, elapsed))
        else:
            print(f"{name}: 路径长度 = {cost:.2f}, 无最优参考, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, None, None, elapsed))

    print(
        f"平均路径长度: {np.mean(total_costs):.2f}, "
        f"平均运行时间: {np.mean(total_times):.3f}s"
    )
    return -np.mean(total_costs)
'''

# =========================================================
# 1) 评估：计算路径长度（最后会自动闭环）
# =========================================================
def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    """
    route 不含重复首尾；本函数会自动把最后一个城市连回起点。
    """
    return (
        sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
        + distances[route[-1], route[0]]
    )

# =========================================================
# 2) 核心演化策略：最便宜插入
# =========================================================
@funsearch.evolve
def tsp_priority(
    current_path: list[int],          # 目前已形成的回路（首尾不同）
    unvisited: list[int],             # 仍待插入的城市
    distances: np.ndarray,            # 距离矩阵
) -> tuple[int, int]:
    """
    选择 (city_to_insert, insert_position)，使新增距离增量最小。
    insert_position 表示要把 city 插到 current_path 的哪个索引之前。
    """
    best_delta = float("inf")
    best_city, best_position = unvisited[0], 1

    path_len = len(current_path)
    for candidate_city in unvisited:
        # 枚举 current_path 上的每条边 (i -> j)
        for idx in range(path_len):
            i = current_path[idx]
            j = current_path[(idx + 1) % path_len]  # % 保证最后连回起点
            delta = (
                distances[i, candidate_city]
                + distances[candidate_city, j]
                - distances[i, j]
            )
            if delta < best_delta:
                best_delta = delta
                best_city = candidate_city
                best_position = idx + 1          # 插到 j 前面

    return best_city, best_position

# =========================================================
# 3) 求解器：使用最便宜插入构造整条路线
# =========================================================
def tsp_solve(distances: np.ndarray, start_city: int = 0) -> list[int]:
    """
    返回一个不含重复首尾的城市顺序；tsp_evaluate 会自动闭环。
    """
    num_cities = distances.shape[0]

    # --- ① 初始回路：任选两个城市 ---
    current_path = [start_city, (start_city + 1) % num_cities]
    unvisited = [c for c in range(num_cities) if c not in current_path]

    # --- ② 逐步插入剩余城市 ---
    while unvisited:
        city_to_insert, insert_pos = tsp_priority(current_path, unvisited, distances)
        current_path.insert(insert_pos, city_to_insert)
        unvisited.remove(city_to_insert)

    return current_path  # 不含重复首尾

# =========================================================
# 4) funsearch 评估入口：保持原有打印与统计格式
# =========================================================
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

        opt_route = instance.get("optimal_tour")
        if compare_with_optimal and opt_route:
            opt_cost = tsp_evaluate(opt_route, distances)
            approx_ratio = cost / opt_cost
            print(
                f"{name}: 路径长度 = {cost:.2f}, "
                f"最优 = {opt_cost:.2f}, 近似比 = {approx_ratio:.4f}, "
                f"时间 = {elapsed:.3f}s"
            )
            summary.append((name, cost, opt_cost, approx_ratio, elapsed))
        else:
            print(f"{name}: 路径长度 = {cost:.2f}, 无最优参考, 时间 = {elapsed:.3f}s")
            summary.append((name, cost, None, None, elapsed))

    print(
        f"平均路径长度: {np.mean(total_costs):.2f}, "
        f"平均运行时间: {np.mean(total_times):.3f}s"
    )
    return -np.mean(total_costs)
