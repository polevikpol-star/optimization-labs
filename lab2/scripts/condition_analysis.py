import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods.gd import GD
from oracles.base import Oracle, OracleInfo
from oracles.quadratic import QuadraticOracle


class MarkowitzConditionOracle(Oracle):

    def __init__(self, covariance, expected_returns):
        self.A = np.asarray(covariance, dtype=float)
        self.b = np.asarray(expected_returns, dtype=float)
        self.dim = len(self.b)
        self.name = "markowitz_condition_experiment"

        eigenvalues = np.linalg.eigvalsh(self.A)
        if eigenvalues[0] <= 0:
            raise ValueError("Ковариационная матрица должна быть положительно определенной")

        self._L = float(eigenvalues[-1])
        self._mu = float(eigenvalues[0])
        self._x_star = np.linalg.solve(self.A, self.b)
        self._f_star = self.f(self._x_star)

    def f(self, x):
        return 0.5 * x @ self.A @ x - self.b @ x

    def grad(self, x):
        return self.A @ x - self.b

    def hess(self, x):
        return self.A

    @property
    def info(self):
        return OracleInfo(L=self._L, mu=self._mu, x_star=self._x_star, f_star=self._f_star)


def main():
    base = QuadraticOracle(dim=2, n_periods=300, seed=0)
    eigvals, eigvecs = np.linalg.eigh(base.A)
    L = float(eigvals[-1])
    base_kappa = L / float(eigvals[0])

    target_kappas = [base_kappa, 10.0, 50.0, 100.0]
    tolerance = 1e-8
    max_iterations = 3000
    x0 = np.zeros(base.dim)
    output_dir = Path(__file__).resolve().parents[1] / "results" / "condition_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Исходная матрица ковариаций:\n", base.A)
    print("Исходные ожидаемые доходности:", base.b)
    print("Начальное число обусловленности: {:.4f}".format(base_kappa))
    print("Шаг GD: {:.6f}".format(1.0 / L))

    results = []
    histories = []

    for i, target_kappa in enumerate(target_kappas):
        if i == 0:
            covariance = base.A.copy()
        else:
            new_eigenvalues = np.array([L / target_kappa, L])
            covariance = eigvecs @ np.diag(new_eigenvalues) @ eigvecs.T
            covariance = 0.5 * (covariance + covariance.T)

        oracle = MarkowitzConditionOracle(covariance, base.b)
        actual_eigenvalues = np.linalg.eigvalsh(oracle.A)
        actual_kappa = float(actual_eigenvalues[-1] / actual_eigenvalues[0])

        method = GD(oracle, step=1.0 / L, max_iter=max_iterations, tol=tolerance)
        history = method.run(x0)
        iterations = len(history.f) - 1  # Первая запись - начальная точка, не шаг.
        converged = history.grad_norm[-1] <= tolerance
        gap = max(0.0, history.f[-1] - oracle.info.f_star)

        results.append({"kappa": actual_kappa,
                        "lambda_min": float(actual_eigenvalues[0]),
                        "lambda_max": float(actual_eigenvalues[-1]),
                        "step": 1.0 / L,
                        "iterations": iterations,
                        "final_grad_norm": history.grad_norm[-1],
                        "final_function_gap": gap,
                        "converged": converged})
        histories.append((actual_kappa, history))
        print("kappa={:.2f} | итераций={} | норма градиента={:.3e} | достигнута точность={}".format(actual_kappa, iterations, history.grad_norm[-1], converged))

    csv_path = output_dir / "condition_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    plt.figure(figsize=(9, 5.5))
    for kappa, history in histories:
        plt.semilogy(np.arange(len(history.grad_norm)), np.maximum(history.grad_norm, 1e-18), label="κ = {:.2f}".format(kappa))
    plt.axhline(tolerance, linestyle="--", linewidth=1.5, label="Порог 1e-8")
    plt.xlabel("Итерация GD")
    plt.ylabel("Норма градиента (логарифмическая шкала)")
    plt.title("Влияние обусловленности на сходимость GD")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "condition_convergence.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    x_labels = ["{:.2f}".format(result["kappa"]) for result in results]
    heights = [result["iterations"] for result in results]
    bars = plt.bar(x_labels, heights)
    plt.bar_label(bars, padding=4)
    plt.xlabel("Число обусловленности κ")
    plt.ylabel("Количество итераций GD")
    plt.title("Итерации до достижения нормы градиента ≤ 1e-8")
    plt.ylim(0, max(heights) * 1.17)
    plt.tight_layout()
    plt.savefig(output_dir / "condition_iterations.png", dpi=180)
    plt.close()

    print("\nРезультаты сохранены в:", output_dir)


if __name__ == "__main__":
    main()
