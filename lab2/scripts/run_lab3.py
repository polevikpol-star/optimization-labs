import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from oracles.constrained_markowitz import ConstrainedMarkowitzOracle
from methods.projected_gd import ProjectedGradientDescent

# 1. Инициализируем оракул 
oracle = ConstrainedMarkowitzOracle(dim=2, n_periods=300, seed=42, lam=1.0, w_max=1.0)

# 2. Запускаем Projected GD
w0 = np.ones(2) / 2  
pgd = ProjectedGradientDescent(oracle=oracle, tol=1e-8)
w_gd, history_w, history_dist = pgd.optimize(w0)

print(f"Итоговые веса (Projected GD): {w_gd}")

# 3. Строим графики
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# График 1: Сходимость
axes[0].plot(history_dist, linewidth=2, color='purple')
axes[0].set_yscale('log')
axes[0].set_title('Сходимость: Изменение весов', fontsize=14)
axes[0].set_xlabel('Итерация')
axes[0].set_ylabel('Ошибка ||w_k - w_{k-1}||')
axes[0].grid(True, which="both", ls="--")

# График 2: Траектория
x = np.linspace(-0.1, 1.1, 100)
y = np.linspace(-0.1, 1.1, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = oracle.f(np.array([X[i,j], Y[i,j]]))

axes[1].contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
axes[1].plot([0, 1], [1, 0], 'k--', label='Бюджет: w1+w2=1', linewidth=2)
axes[1].fill_between([0, 1], [1, 0], [0, 0], color='gray', alpha=0.2, label='Допустимая область')

history_w = np.array(history_w)
axes[1].plot(history_w[:, 0], history_w[:, 1], 'ro-', markersize=4, label='Траектория PGD')

axes[1].set_title('Траектория Projected GD', fontsize=14)
axes[1].set_xlabel('Вес w1')
axes[1].set_ylabel('Вес w2')
axes[1].legend()
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()
