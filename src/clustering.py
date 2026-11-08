import numpy as np
from typing import List
from .hull_checks import any_majority_inside

def build_safe_clusters(X_minority: np.ndarray, X_majority: np.ndarray) -> List[np.ndarray]:
    """
    Жадно объединяет близкие точки меньшинства в кластеры при условии,
    что их выпуклая оболочка не содержит точек большинство.
    """
    n_samples = len(X_minority)
    clusters = [[i] for i in range(n_samples)]
    
    changed = True
    while changed:
        changed = False
        n_clusters = len(clusters)
        best_pair = None
        min_dist = np.inf
        
        # Ищем наиближайшие кластеры для объединения
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                pts_i = X_minority[clusters[i]]
                pts_j = X_minority[clusters[j]]
                
                # Расстояние между центроидами
                dist = np.linalg.norm(np.mean(pts_i, axis=0) - np.mean(pts_j, axis=0))
                
                if dist < min_dist:
                    cand_pts = np.vstack([pts_i, pts_j])
                    if not any_majority_inside(cand_pts, X_majority):
                        min_dist = dist
                        best_pair = (i, j)
                        
        if best_pair is not None:
            i, j = best_pair
            clusters[i] = clusters[i] + clusters[j]
            clusters.pop(j)
            changed = True
            
    return [X_minority[idx_list] for idx_list in clusters]