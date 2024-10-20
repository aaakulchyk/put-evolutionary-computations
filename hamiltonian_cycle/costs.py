import itertools as it
import math

import numpy as np
import pandas as pd


def nodes_euclidean_distance(a: pd.Series, b: pd.Series) -> int:
    return np.int32(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) + 0.5)


def nodes_cost(a: pd.Series, b: pd.Series) -> int:
    return a["cost"] + b["cost"]


def function_cost(ds: pd.DataFrame) -> int:
    cost = ds["cost"].sum()
    for i in range(-1, len(ds) - 1):
        cost += nodes_euclidean_distance(ds.iloc[i], ds.iloc[i + 1])
    return int(cost)


def dm(ds: pd.DataFrame) -> np.ndarray:
    n_nodes = len(ds)
    dm = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in it.product(range(n_nodes), range(n_nodes)):
        if i == j:
            continue
        a, b = ds.iloc[i], ds.iloc[j]
        dm[i, j] = nodes_euclidean_distance(a, b) + nodes_cost(a, b)
    return dm
