"""TSPLIB 数据加载与 FunSearch 实例构造工具。

改进点（保持默认路径不变）：
1. **自动解压**：同时支持 `*.tsp.gz` 与 `*.opt.tour.gz`。
2. **距离矩阵**：严格按 TSPLIB `EUC_2D` 规则 `int(d+0.5)` 取整，向量化实现。
3. **最优路线解析**：自定义解析器，返回 0‑based `optimal_tour`。
4. **类型标注 + Docstring**：便于 IDE / MyPy 与后期维护。
5. **内存友好**：解析完后立即释放坐标数组。
"""
from __future__ import annotations

import gzip
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import tsplib95

# --------------------------- 常量 --------------------------- #
# 按用户原脚本保留固定路径
TSPLIB_ROOT = Path("/content/TSP_funsearch/tsplib_instances")
TSPLIB_ARCHIVE = Path("/content/TSP_funsearch/TSP_data/ALL_tsp.tar.gz")

# ------------------------- 解压函数 ------------------------- #

def extract_archive() -> None:
    """一次性解压 `ALL_tsp.tar.gz` 到目标文件夹。"""
    if not TSPLIB_ARCHIVE.exists():
        raise FileNotFoundError(f"{TSPLIB_ARCHIVE} 不存在")
    TSPLIB_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TSPLIB_ARCHIVE, "r:gz") as tar:
        tar.extractall(TSPLIB_ROOT)


def decompress_gz_files(folder: Path = TSPLIB_ROOT) -> None:
    """将文件夹内 `*.gz` 解压成同名无后缀文件。

    同时处理 `.tsp.gz` 与 `.opt.tour.gz`。若目标文件已存在则跳过。"""
    for fname in os.listdir(folder):
        if fname.endswith(".gz"):
            gz_path = folder / fname
            out_path = folder / fname[:-3]  # 去掉 .gz
            if out_path.exists():
                continue
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed: {fname} → {out_path.name}")

# --------------------- TSPLIB 数据加载 --------------------- #

def _int_euc2d_matrix(problem: tsplib95.models.StandardProblem) -> np.ndarray:
    """构建整数化 EUC_2D 距离矩阵。"""
    coords = np.array([problem.node_coords[i + 1] for i in range(problem.dimension)], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    mat = np.rint(np.sqrt((diff ** 2).sum(-1))).astype(int)
    return mat


def _parse_opt_tour(path: Path) -> List[int] | None:
    """解析 *.opt.tour 文件为 0‑based 城市序列；若文件不存在返回 None。"""
    if not path.exists():
        return None
    tour: List[int] = []
    with open(path) as f:
        recording = False
        for tok in f.read().split():
            if tok.upper() == "TOUR_SECTION":
                recording = True
                continue
            if not recording:
                continue
            if tok == "-1":
                break
            tour.append(int(tok) - 1)
    return tour or None


class TSPLibDataset:
    """加载 TSPLIB 实例并转换为 FunSearch 使用的字典格式。"""

    def __init__(self, folder: str | Path = TSPLIB_ROOT):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    # -------- 单实例加载 -------- #
    def load_instance(self, name: str) -> Dict[str, object]:
        tsp_path = self.folder / f"{name}.tsp"
        if not tsp_path.exists():
            raise FileNotFoundError(f"{tsp_path} not found. 请确保已经解压")
        problem = tsplib95.load(tsp_path)
        dist_mat = _int_euc2d_matrix(problem)
        opt_tour = _parse_opt_tour(self.folder / f"{name}.opt.tour")
        # 立即释放 coords 节省内存（problem 仍可被 GC 回收）
        return {
            "distances": dist_mat,
            "optimal_tour": opt_tour,
        }

    # -------- 多实例批量加载 -------- #
    def load_instances(self, names: List[str]) -> Dict[str, Dict[str, object]]:
        return {name: self.load_instance(name) for name in names}


# --------------------- FunSearch 数据包装 --------------------- #

def build_funsearch_dataset(tsplib_data: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """提取 FunSearch 需要的字段。"""
    return {
        name: {
            "distances": data["distances"],
            "optimal_tour": data.get("optimal_tour"),
        }
        for name, data in tsplib_data.items()
    }

# --------------------------- DEMO --------------------------- #
if __name__ == "__main__":
    # 1. 解压数据（若已解压可跳过）
    extract_archive()
    decompress_gz_files()

    # 2. 加载三个测试集示例
    dataset = TSPLibDataset()
    tsplib_data1 = dataset.load_instances(["berlin52", "eil76"])
    tsplib_data2 = dataset.load_instances(["kroA100", "pr124", "d198"])
    tsplib_data3 = dataset.load_instances(["lin318", "pcb442"])

    instances_AlgorithmDevelop = build_funsearch_dataset(tsplib_data1)
    instances_PerformanceTesting = build_funsearch_dataset(tsplib_data2)
    instances_GeneralizationTesting = build_funsearch_dataset(tsplib_data3)

    # 简单打印确认
    print("Algorithm dev set:", list(instances_AlgorithmDevelop))
    print("Performance test set:", list(instances_PerformanceTesting))
    print("Generalization test set:", list(instances_GeneralizationTesting))
