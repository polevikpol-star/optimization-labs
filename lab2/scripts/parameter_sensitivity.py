import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods.gd import GD
from oracles.quadratic import QuadraticOracle
from scripts.condition_analysis import MarkowitzConditionOracle


TOL = 1e-8
MAX_ITER = 5000


def run_gd(oracle, start, step):
    history = GD(oracle, step=step, max_iter=MAX_ITER, tol=TOL).run(start)
    x_final = np.asarray(history.x[-1])
    converged = float(history.grad_norm[-1]) <= TOL
    return history, x_final, converged


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lambda_experiment(sigma, mu, step, output_dir):
    rows = []
    exact_points = []
    for lam in [0.5, 1.0, 2.0]:
        oracle = MarkowitzConditionOracle(sigma, lam * mu)
        x_star = oracle.info.x_star
        history, x_final, converged = run_gd(oracle, np.zeros(2), step)
        rows.append({"lambda": lam,
                     "kappa": float(np.linalg.cond(sigma)),
                     "w1_exact": float(x_star[0]),
                     "w2_exact": float(x_star[1]),
                     "w1_gd": float(x_final[0]),
                     "w2_gd": float(x_final[1]),
                     "iterations": len(history.f) - 1,
                     "final_grad_norm": float(history.grad_norm[-1]),
                     "distance_to_exact": float(np.linalg.norm(x_final - x_star)),
                     "converged": converged})
        exact_points.append(x_star)
        assert converged, "GD не достиг точности при lambda={}".format(lam)
        assert np.linalg.norm(x_final - x_star) <= 1.1 * TOL / oracle.info.mu
    write_csv(output_dir / "lambda_results.csv", rows)

    np.testing.assert_allclose(exact_points[2], 2 * exact_points[1], rtol=1e-10)
    np.testing.assert_allclose(exact_points[0], .5 * exact_points[1], rtol=1e-10)

    plt.figure(figsize=(8, 5))
    lambdas = [r["lambda"] for r in rows]
    plt.plot(lambdas, [r["w1_exact"] for r in rows], "o-", label="w1: точный оптимум")
    plt.plot(lambdas, [r["w2_exact"] for r in rows], "s-", label="w2: точный оптимум")
    plt.scatter(lambdas, [r["w1_gd"] for r in rows], marker="x", s=70, label="GD, w1")
    plt.scatter(lambdas, [r["w2_gd"] for r in rows], marker="+", s=90, label="GD, w2")
    plt.xlabel("Коэффициент важности доходности lambda")
    plt.ylabel("Оптимальные позиции активов")
    plt.title("Зависимость оптимальных позиций от lambda")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "lambda_sensitivity.png", dpi=180)
    plt.close()
    return rows


def initial_point_experiment(sigma, mu, step, output_dir):
    oracle = MarkowitzConditionOracle(sigma, mu)
    starts = [("(0, 0)", np.array([0., 0.])),
              ("(5, -5)", np.array([5., -5.])),
              ("(-5, 5)", np.array([-5., 5.])),
              ("(20, -10)", np.array([20., -10.]))]
    rows, histories = [], []
    for label, start in starts:
        history, x_final, converged = run_gd(oracle, start, step)
        rows.append({"start": label,
                     "start_w1": float(start[0]),
                     "start_w2": float(start[1]),
                     "w1_exact": float(oracle.info.x_star[0]),
                     "w2_exact": float(oracle.info.x_star[1]),
                     "w1_gd": float(x_final[0]),
                     "w2_gd": float(x_final[1]),
                     "iterations": len(history.f) - 1,
                     "final_grad_norm": float(history.grad_norm[-1]),
                     "distance_to_exact": float(np.linalg.norm(x_final - oracle.info.x_star)),
                     "converged": converged})
        histories.append((label, np.asarray(history.x), history))
        assert converged, "GD не достиг точности при w0={}".format(label)
        assert np.linalg.norm(x_final - oracle.info.x_star) <= 1.1 * TOL / oracle.info.mu
    write_csv(output_dir / "initial_points_results.csv", rows)

    all_points = np.vstack([np.asarray(oracle.info.x_star)[None, :]] + [p for _, p, _ in histories])
    xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
    ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()
    padding_x = max(2.0, 0.14 * (xmax - xmin))
    padding_y = max(2.0, 0.14 * (ymax - ymin))
    xx, yy = np.meshgrid(np.linspace(xmin-padding_x, xmax+padding_x, 280),
                         np.linspace(ymin-padding_y, ymax+padding_y, 280))
    dx = xx - oracle.info.x_star[0]
    dy = yy - oracle.info.x_star[1]
    gap = .5 * (sigma[0, 0] * dx**2 + 2 * sigma[0, 1] * dx * dy + sigma[1, 1] * dy**2)
    positive_max = float(gap.max())
    levels = np.geomspace(positive_max * 1e-4, positive_max * 0.85, 10)

    plt.figure(figsize=(9, 7))
    cs = plt.contour(xx, yy, gap, levels=levels, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
    for label, points, history in histories:
        plt.plot(points[:, 0], points[:, 1], linewidth=2,
                 label="Старт {}: {} шагов".format(label, len(history.f)-1))
        plt.scatter(points[0, 0], points[0, 1], s=48)
    plt.scatter(oracle.info.x_star[0], oracle.info.x_star[1], marker="*", s=190,
                color="black", zorder=5, label="Точный минимум w*")
    plt.xlabel("Позиция w1")
    plt.ylabel("Позиция w2")
    plt.title("GD из разных начальных точек на линиях уровня f(w) - f*")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "initial_trajectories.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for label, _, history in histories:
        plt.semilogy(np.arange(len(history.grad_norm)), history.grad_norm,
                     label="Старт {}".format(label))
    plt.axhline(TOL, linestyle="--", color="black", label="Порог 1e-8")
    plt.xlabel("Итерация GD")
    plt.ylabel("Норма градиента (логарифмическая шкала)")
    plt.title("Зависимость сходимости GD от начальной точки")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "initial_convergence.png", dpi=180)
    plt.close()
    return rows

def main():
    base = QuadraticOracle(dim=2, n_periods=300, seed=0)
    sigma = base.A.copy()
    mu = base.b.copy()
    eigvals = np.linalg.eigvalsh(sigma)
    L, m = float(eigvals[-1]), float(eigvals[0])
    assert m > 0, "Sigma должна быть положительно определенной"
    step = 1.0 / L
    output_dir = Path(__file__).resolve().parents[1] / "results" / "parameter_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Данные из QuadraticOracle команды: dim=2, n_periods=300, seed=0")
    print("Собственные значения:", eigvals)
    print("Число обусловленности: {:.4f}; шаг GD: {:.6f}".format(L/m, step))

    print("\n1. ВЛИЯНИЕ ПАРАМЕТРА lambda")
    for row in lambda_experiment(sigma, mu, step, output_dir):
        print("lambda={:.1f} | w* = ({:.4f}, {:.4f}) | шагов={} | сошелся={}".format(
            row["lambda"], row["w1_exact"], row["w2_exact"], row["iterations"], row["converged"]))

    print("\n2. ВЛИЯНИЕ НАЧАЛЬНОЙ ТОЧКИ")
    for row in initial_point_experiment(sigma, mu, step, output_dir):
        print("w0={} | шагов={} | расстояние до w*={:.3e} | сошелся={}".format(
            row["start"], row["iterations"], row["distance_to_exact"], row["converged"]))

    print("\nТаблицы и графики сохранены:", output_dir)


if __name__ == "__main__":
    main()
