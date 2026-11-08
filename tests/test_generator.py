import numpy as np
from src.generator import extend_smaller_class

def test_extend_smaller_class_shape():
    X_min = np.random.rand(10, 2)
    X_maj = np.random.rand(50, 2) + 2.0  # Рассогласованные кластеры
    
    X_min_balanced = extend_smaller_class(X_min, X_maj, target_count=50)
    assert len(X_min_balanced) == 50