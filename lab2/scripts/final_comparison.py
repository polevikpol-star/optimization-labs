"""Итоговое сравнение всех подходов на одной и той же задаче Марковица.

Что делает:
  A) dim=2: GD, Nesterov, адаптивные шаги, L2-регуляризация.
  B) dim=50: GD, Nesterov, адаптивные шаги.
Для всех методов: одинаковая задача, x0 = 0, критерий ||grad|| <= TOL.

Результаты (папка results/final/):
  summary_dim2.csv / summary_dim50.csv  - таблицы
  summary.md                            - те же таблицы в markdown (для README)
  convergence_dim2.png, convergence_dim50.png - f(x_k) - f*
  iterations_dim2.png, iterations_dim50.png   - сравнение числа итераций
  regularization_tradeoff.png                 - итерации vs потеря качества
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods import GD, AdaptiveStep, Nesterov
from oracles.quadratic import QuadraticOracle
from oracles.regularized_markowitz import RegularizedMarkowitzOracle
from utils.metrics import Timer

TOL = 1e-8
MAX_ITER = 200000
OUT = Path(__file__).resolve().parents[1] / "results" / "final"


def run_method(name, method, oracle, base_oracle, x0):
    """Запускает метод и считает метрики. Качество всегда измеряем на исходной функции."""
    with Timer() as t:
        h = method.run(x0)
    x = h.x[-1]
    gap = max(0.0, float(base_oracle.f(x) - base_oracle.info.f_star))  # шум ~1e-18 -> 0
    return h, {
        "method": name,
        "iterations": len(h.f) - 1,
        "converged": bool(h.grad_norm[-1] <= TOL),
        "time_ms": round(t.elapsed * 1e3, 3),
        "final_grad_norm": float(h.grad_norm[-1]),
        "gap_f": gap,
        "rel_loss": gap / abs(base_oracle.info.f_star),
        "dist_to_x_star": float(np.linalg.norm(x - base_oracle.info.x_star)),
    }


def build_methods(oracle):
    L, mu = oracle.info.L, oracle.info.mu
    kw = dict(max_iter=MAX_ITER, tol=TOL)
    runs = {
        "GD, шаг 1/L (baseline)": GD(oracle, step=1.0 / L, **kw),
        "GD, шаг 2/(L+mu)": GD(oracle, step=2.0 / (L + mu), **kw),
        "Nesterov, шаг 1/L": Nesterov(oracle, step=1.0 / L, **kw),
    }
    for s, label in [("polyak", "GD + Polyak step"),
                     ("bb1", "GD + Barzilai-Borwein 1"),
                     ("bb2", "GD + Barzilai-Borwein 2"),
                     ("backtrack", "GD + Armijo backtracking")]:
        runs[label] = AdaptiveStep(oracle, strategy=s, eta0=1.0 / L, **kw)
    return runs


def save_table(rows, path):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, bool):
                v = "да" if v else "нет"
            elif isinstance(v, float):
                v = f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) >= 1e5) else f"{v:.3g}"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def experiment(dim, tag, with_regularization):
    base = QuadraticOracle(dim=dim, n_periods=300, seed=0)
    kappa = base.info.L / base.info.mu
    x0 = np.zeros(base.dim)
    print(f"\n=== {tag}: dim={dim}, kappa={kappa:.2f} ===")

    rows, hists = [], {}
    for name, m in build_methods(base).items():
        h, row = run_method(name, m, base, base, x0)
        row["kappa"] = kappa
        rows.append(row)
        hists[name] = h
        print(f"{name:<28} iters={row['iterations']:>7} gap={row['gap_f']:.2e} "
              f"conv={row['converged']}")

    reg_rows = []
    if with_regularization:
        for alpha in [1e-6, 1e-5, 1e-4, 1e-3]:
            ro = RegularizedMarkowitzOracle(base, alpha)
            eta = 2.0 / (ro.info.L + ro.info.mu)
            m = GD(ro, step=eta, max_iter=MAX_ITER, tol=TOL)
            name = f"GD + L2 (alpha={alpha:g})"
            h, row = run_method(name, m, ro, base, x0)
            # у регуляризованной задачи свой оптимум: считаем сходимость по ней,
            # а потерю качества - на исходной функции
            row["gap_f"] = float(base.f(h.x[-1]) - base.info.f_star)
            row["kappa"] = ro.info.L / ro.info.mu
            row["rel_loss"] = row["gap_f"] / abs(base.info.f_star)
            row["dist_to_x_star"] = float(np.linalg.norm(h.x[-1] - base.info.x_star))
            rows.append(row)
            reg_rows.append(row)
            hists[name] = h
            print(f"{name:<28} iters={row['iterations']:>7} gap={row['gap_f']:.2e} "
                  f"kappa={row['kappa']:.2f}")

    # ---- графики ----
    f_star = base.info.f_star
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, h in hists.items():
        if name.startswith("GD + L2"):
            continue  # у другой задачи другой f*, на этом графике не рисуем
        gap = np.maximum(np.array(h.f) - f_star, 1e-20)
        ax.semilogy(gap, label=name)
    ax.set_xscale("symlog")
    ax.set_xlabel("итерация (symlog)")
    ax.set_ylabel("f(w_k) - f*")
    ax.set_title(f"Сходимость методов, dim={dim}, κ≈{kappa:.1f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"convergence_{tag}.png", dpi=170)
    plt.close(fig)

    bar_rows = [r for r in rows if not r["method"].startswith("GD + L2")]
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["method"] for r in bar_rows]
    iters = [r["iterations"] for r in bar_rows]
    bars = ax.barh(names, iters, color="#4C78A8")
    ax.bar_label(bars, padding=3)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel(f"число итераций до ||grad|| ≤ {TOL:g} (log)")
    ax.set_title(f"Сравнение числа итераций, dim={dim}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"iterations_{tag}.png", dpi=170)
    plt.close(fig)

    save_table(rows, OUT / f"summary_{tag}.csv")
    return rows, reg_rows


def regularization_plot(base_row, reg_rows):
    """Главный компромисс регуляризации: меньше итераций, но хуже исходная функция"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    alphas = [0.0] + [float(r["method"].split("=")[1].rstrip(")")) for r in reg_rows]
    iters = [base_row["iterations"]] + [r["iterations"] for r in reg_rows]
    gaps = [0.0] + [r["gap_f"] for r in reg_rows]
    x = np.arange(len(alphas))
    labels = ["0"] + [f"{a:g}" for a in alphas[1:]]
    ax1.bar(x, iters, color="#4C78A8", width=0.5)
    ax1.set_xticks(x, labels)
    ax1.set_xlabel("alpha (L2-регуляризация)")
    ax1.set_ylabel("итерации GD", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(x, gaps, "o-", color="#E45756")
    ax2.set_ylabel("f_исх(w) - f*_исх  (потеря качества)", color="#E45756")
    ax1.set_title("Регуляризация: быстрее, но решение хуже")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "regularization_tradeoff.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows2, reg2 = experiment(2, "dim2", with_regularization=True)
    rows50, _ = experiment(50, "dim50", with_regularization=False)

    base_row = next(r for r in rows2 if r["method"] == "GD, шаг 2/(L+mu)")
    regularization_plot(base_row, reg2)

    cols = ["method", "kappa", "iterations", "time_ms", "gap_f",
            "rel_loss", "dist_to_x_star", "converged"]
    heads = ["Метод", "κ задачи", "Итерации", "Время, мс",
             "f−f* (исходная)", "(f−f*)/abs(f*)", "‖w−w*‖", "Сошёлся"]
    md = ["## Задача A: dim=2\n", md_table(rows2, cols, heads), "",
          "## Задача B: dim=50\n", md_table(rows50, cols, heads), ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nФайлы сохранены в:", OUT)


if __name__ == "__main__":
    main()
