import os
import numpy as np
import matplotlib.pyplot as plt

from oracles import QuadraticOracle
from methods import GD, AdaptiveStep

# одна и та же задача и одна и та же стартовая точка для всех методов
oracle = QuadraticOracle(dim=50, n_periods=300, seed=0)
f_star = oracle.info.f_star
L = oracle.info.L
x0 = np.zeros(oracle.dim)
tol, max_iter = 1e-8, 20000

print(f"kappa = {oracle.info.L / oracle.info.mu:.1f}")

runs = {"gd (шаг 1/L)": GD(oracle, step=1.0 / L, max_iter=max_iter, tol=tol)}
for s in ["polyak", "bb1", "bb2", "backtrack"]:
    runs[s] = AdaptiveStep(oracle, strategy=s, eta0=1.0 / L,
                           max_iter=max_iter, tol=tol)

print(f"{'метод':<14}{'итераций':>10}{'f - f*':>14}{'||grad||':>14}")
for name, m in runs.items():
    h = m.run(x0)
    gap = np.maximum(np.array(h.f) - f_star, 1e-20)
    print(f"{name:<14}{len(h.f) - 1:>10}{gap[-1]:>14.2e}{h.grad_norm[-1]:>14.2e}")
    plt.semilogy(gap, label=name)

os.makedirs("results", exist_ok=True)
plt.xlabel("итерация")
plt.ylabel("f(x_k) - f*")
plt.title("GD и адаптивные шаги")
plt.legend()
plt.grid(True)
plt.savefig("results/compare_adaptive.png", dpi=150)
print("График сохранён: results/compare_adaptive.png")