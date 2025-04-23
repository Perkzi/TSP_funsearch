# pip install tsplib95 大前提
import tarfile
import os
import numpy as np
import tsplib95
import matplotlib.pyplot as plt
import pandas as pd
with tarfile.open("/content/ALL_tsp.tar.gz", "r:gz") as tar:
    tar.extractall("/content/TSP_funsearch/tsplib_instances")
import gzip
import shutil
import os
# 解压缩
def decompress_gz_files(folder="/content/TSP_funsearch/tsplib_instances"):
    for fname in os.listdir(folder):
        if fname.endswith(".tsp.gz"):
            gz_path = os.path.join(folder, fname)
            tsp_path = os.path.join(folder, fname[:-3])  # remove .gz
            if not os.path.exists(tsp_path):
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(tsp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Decompressed: {fname} → {fname[:-3]}")

decompress_gz_files("/content/TSP_funsearch/tsplib_instances")
# ========== 数据加载器 + 构造 FunSearch 格式测试集 ==========
os
import gzip
import numpy as np
import tsplib95
import funsearch

class TSPLibDataset:
    def __init__(self, folder="/content/TSP_funsearch/tsplib_instances"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def load_instance(self, name: str):
        tsp_path = os.path.join(self.folder, f"{name}.tsp")
        tour_path = os.path.join(self.folder, f"{name}.opt.tour")
        if not os.path.exists(tsp_path):
            raise FileNotFoundError(f"{tsp_path} not found. Please ensure it exists.")
        problem = tsplib95.load(tsp_path)
        coords = [problem.node_coords[i + 1] for i in range(problem.dimension)]
        n = len(coords)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))

        opt_tour = None
        if os.path.exists(tour_path):
            try:
                opt = tsplib95.load(tour_path)
                opt_tour = list(opt.tours[0])
            except:
                print(f"Warning: Failed to load optimal tour from {tour_path}")

        return {
            "name": name,
            "dimension": n,
            "distances": distances,
            "coords": coords,
            "optimal_tour": opt_tour
        }

    def load_instances(self, names: list[str]) -> dict:
        return {name: self.load_instance(name) for name in names}


def build_funsearch_dataset(tsplib_data: dict) -> dict:
    """
    tsplib_data: {'berlin52': {'distances': ndarray, 'optimal_tour': list, ...}, ...}
    返回值必须是 dict[name] -> {"distances": ndarray, "optimal_tour": Optional[list]}
    """
    return {
        name: {
            "distances": data["distances"],
            "optimal_tour": data.get("optimal_tour")
        }
        for name, data in tsplib_data.items()
    }


dataset = TSPLibDataset("/content/TSP_funsearch/tsplib_instances")
tsplib_data1 = dataset.load_instances(["berlin52", "eil76"]) 
tsplib_data2 = dataset.load_instances(["kroA100", "pr124","d198" ])
tsplib_data3 = dataset.load_instances(["lin318", "pcb442"])
# --------在笔记本里调用------
# instances_AlgorithmDevelop = build_funsearch_dataset(tsplib_data1)
# instances_PerformanceTesting = build_funsearch_dataset(tsplib_data2)
# instances_GeneralizationTesting = build_funsearch_dataset(tsplib_data3)

printf("在笔记本里调用: instances_AlgorithmDevelop = build_funsearch_dataset(tsplib_data1)")
printf("在笔记本里调用: instances_PerformanceTesting = build_funsearch_dataset(tsplib_data2)")
printf("在笔记本里调用: instances_GeneralizationTesting = build_funsearch_dataset(tsplib_data3)")
# ------------- examples --------------- #
# --- Algorithm Development --- #
# berlin52 52 EUC_2D Yes Standard benchmark with Euclidean distances
# eil76 76 EUC_2D Yes Small-sized, regularly structured instance 
# --- Performance Testing --- #
# kroA100 100 EUC_2D Yes Classic mid-size instance
# pr124 124 EUC_2D Yes Good for testing generalization
# d198 198 EUC_2D Yes Mid-sized, realistic urban layout
# --- Generalization Testing --- #
# lin318 318 EUC_2D No No optimal tour available; good for large-scale testing
# pcb442 442 EUC_2D Yes Complex city distribution; useful for robustness testing

