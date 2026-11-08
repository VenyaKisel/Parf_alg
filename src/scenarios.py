import numpy as np
from sklearn.datasets import make_moons, make_circles

def generate_scenario_data(scenario: str = "moons", n_samples: int = 400, noise: float = 0.1, random_state: int = 42):
    """Генерация различных синтетических геометрий для тестов."""
    np.random.seed(random_state)
    if scenario == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif scenario == "concentric":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif scenario == "separated":
        X1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], n_samples // 2)
        X2 = np.random.multivariate_normal([3, 3], [[0.1, 0], [0, 0.1]], n_samples // 2)
        X = np.vstack([X1, X2])
        y = np.array([0] * len(X1) + [1] * len(X2))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    # Делаем дисбаланс 1:5
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    idx_0_sub = idx_0[:len(idx_0) // 5]
    
    keep_idx = np.concatenate([idx_0_sub, idx_1])
    return X[keep_idx], y[keep_idx]