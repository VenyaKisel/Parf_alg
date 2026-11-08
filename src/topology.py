import numpy as np
import gudhi as gd
from typing import Dict

def compute_betti_numbers(X: np.ndarray, max_edge_length: float = 0.5) -> Dict[str, int]:
    """
    Вычисляет числа Бетти (beta_0, beta_1) для облака точек X
    с помощью комплекса Вьеториса-Рипса (Rips Complex).
    """
    if len(X) == 0:
        return {"betti_0": 0, "betti_1": 0}
    
    # Строим комплекс Вьеториса-Рипса
    rips_complex = gd.RipsComplex(points=X, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # Вычисляем гомологии
    simplex_tree.compute_persistence_pairs()
    betti = simplex_tree.betti_numbers()
    
    b0 = betti[0] if len(betti) > 0 else 0
    b1 = betti[1] if len(betti) > 1 else 0
    
    return {"betti_0": b0, "betti_1": b1}

def get_persistence_diagram(X: np.ndarray, max_edge_length: float = 1.0):
    """Возвращает диаграмму персистентности для визуализации."""
    rips_complex = gd.RipsComplex(points=X, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    return simplex_tree, diag