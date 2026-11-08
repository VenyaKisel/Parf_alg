import numpy as np
from scipy.optimize import linprog

def point_in_convex_hull(point: np.ndarray, hull_points: np.ndarray, tol: float = 1e-7) -> bool:
    """
    Проверяет, лежит ли point внутри выпуклой оболочки hull_points с помощью линейного программирования.
    Решает задачу: sum(alpha_i * x_i) = point, sum(alpha_i) = 1, alpha_i >= 0.
    """
    n_points, dim = hull_points.shape
    if n_points == 0:
        return False
    if n_points == 1:
        return bool(np.allclose(point, hull_points[0], atol=tol))

    # СЛАУ: A_eq @ alpha = b_eq
    A_eq = np.vstack([hull_points.T, np.ones(n_points)])
    b_eq = np.append(point, 1.0)
    
    # Целевая функция нулевая (просто ищем допустимую точку)
    c = np.zeros(n_points)
    bounds = [(0, 1) for _ in range(n_points)]
    
    res = linprog(c, A_eq_ub=None, b_eq_ub=None, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return bool(res.success)

def any_majority_inside(hull_points: np.ndarray, X_majority: np.ndarray, tol: float = 1e-7) -> bool:
    """Проверяет, попадает ли хотя бы одна точка большинства внутрь выпуклой оболочки."""
    if len(hull_points) < 2 or len(X_majority) == 0:
        return False
    for pt in X_majority:
        if point_in_convex_hull(pt, hull_points, tol=tol):
            return True
    return False