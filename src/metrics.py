import numpy as np
from scipy.spatial.distance import cdist

def single_linkage_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Вычисляет минимальное расстояние между двумя множествами точек (Single Linkage)."""
    if len(X1) == 0 or len(X2) == 0:
        return np.inf
    return float(np.min(cdist(X1, X2)))

def hausdorff_distance_custom(X1: np.ndarray, X2: np.ndarray) -> float:
    """Вычисляет направленное расстояние Хаусдорфа от X1 к X2."""
    if len(X1) == 0 or len(X2) == 0:
        return np.inf
    dists = cdist(X1, X2)
    return float(np.max(np.min(dists, axis=1)))