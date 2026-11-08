import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from sklearn.ensemble import RandomForestClassifier

def plot_oversampling_comparison(X_orig: np.ndarray, y_orig: np.ndarray, 
                                 X_bal: np.ndarray, y_bal: np.ndarray, 
                                 title: str = "Сравнение данных"):
    """Отрисовывает 2D-распределение данных ДО и ПОСЛЕ оверсемплинга."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ДО
    axes[0].scatter(X_orig[y_orig==0][:, 0], X_orig[y_orig==0][:, 1], color='blue', label='Меньшинство (0)', alpha=0.7)
    axes[0].scatter(X_orig[y_orig==1][:, 0], X_orig[y_orig==1][:, 1], color='gray', label='Большинство (1)', alpha=0.3)
    axes[0].set_title(f"{title} — Исходные данные")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. ПОСЛЕ
    syn_mask = np.ones(len(X_bal), dtype=bool)
    syn_mask[:len(X_orig)] = False  # Отделяем новые синтетические точки
    
    axes[1].scatter(X_orig[y_orig==1][:, 0], X_orig[y_orig==1][:, 1], color='gray', label='Большинство (1)', alpha=0.3)
    axes[1].scatter(X_orig[y_orig==0][:, 0], X_orig[y_orig==0][:, 1], color='blue', label='Исходное меньшинство', alpha=0.7)
    axes[1].scatter(X_bal[syn_mask & (y_bal==0)][:, 0], X_bal[syn_mask & (y_bal==0)][:, 1], 
                    color='red', marker='x', label='Синтетика (Hull)', alpha=0.8)
    axes[1].set_title(f"{title} — После оверсемплинга")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_decision_boundary_and_homology(X_orig: np.ndarray, y_orig: np.ndarray,
                                        X_bal: np.ndarray, y_bal: np.ndarray,
                                        model=None):
    """
    Показывает, как изменяется разделяющая поверхность классификатора 
    и как точки забивают топологические отверстия.
    """
    if model is None:
        model = RandomForestClassifier(random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Сетка для разделяющей поверхности
    x_min, x_max = X_bal[:, 0].min() - 0.5, X_bal[:, 0].max() + 0.5
    y_min, y_max = X_bal[:, 1].min() - 0.5, X_bal[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Модель ДО
    model.fit(X_orig, y_orig)
    Z1 = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0].contourf(xx, yy, Z1, alpha=0.3, cmap='coolwarm')
    axes[0].scatter(X_orig[:, 0], X_orig[:, 1], c=y_orig, cmap='coolwarm', edgecolors='k')
    axes[0].set_title("Граница решения ДО (Исходный класс)")
    
    # Модель ПОСЛЕ
    model.fit(X_bal, y_bal)
    Z2 = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z2, alpha=0.3, cmap='coolwarm')
    axes[1].scatter(X_bal[:, 0], X_bal[:, 1], c=y_bal, cmap='coolwarm', edgecolors='k')
    axes[1].set_title("Граница решения ПОСЛЕ (Разрушение многообразия)")
    
    plt.tight_layout()
    plt.show()

def plot_persistence_diagrams(diag_before, diag_after):
    """Отрисовка устойчивых гомологий (Persistence Diagrams) Gudhi."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gd.plot_persistence_diagram(diag_before, axes=axes[0])
    axes[0].set_title("Гомологии ДО (Сохранена топология)")
    
    gd.plot_persistence_diagram(diag_after, axes=axes[1])
    axes[1].set_title("Гомологии ПОСЛЕ (Дыры затянуты)")
    
    plt.tight_layout()
    plt.show()