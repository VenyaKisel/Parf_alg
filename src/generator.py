import numpy as np
from .clustering import build_safe_clusters

def generate_points_in_hull(hull_points: np.ndarray, n_samples: int) -> np.ndarray:
    """Генерирует n_samples точек как случайные выпуклые комбинации вершин hull_points."""
    k = len(hull_points)
    if k == 1:
        return np.repeat(hull_points, n_samples, axis=0)
    
    # Семплируем веса Дирихле (выпуклая комбинация)
    weights = np.random.dirichlet(np.ones(k), size=n_samples)
    return weights @ hull_points

def extend_smaller_class(X_minority: np.ndarray, X_majority: np.ndarray, target_count: int) -> np.ndarray:
    """Выполняет оверсемплинг меньшего класса до размера target_count."""
    n_current = len(X_minority)
    if n_current >= target_count:
        return X_minority
    
    n_to_generate = target_count - n_current
    clusters = build_safe_clusters(X_minority, X_majority)
    
    generated = []
    per_cluster = max(1, n_to_generate // len(clusters))
    
    for cluster_pts in clusters:
        syn = generate_points_in_hull(cluster_pts, per_cluster)
        generated.append(syn)
        
    X_synthetic = np.vstack(generated)
    if len(X_synthetic) > n_to_generate:
        X_synthetic = X_synthetic[:n_to_generate]
        
    return np.vstack([X_minority, X_synthetic])