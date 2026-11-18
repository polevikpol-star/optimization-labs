"""ЛР3. Итоговое сравнение методов на задаче Марковица с ограничениями.

Задача: min f(w) = 1/2 wᵀΣw − λ μᵀw  при  Σwᵢ = 1,  0 ≤ wᵢ ≤ w_max,  μᵀw ≥ r_min.
Все методы - проекционные: w_{k+1} = Π_C(w_k − η_k ∇f(w_k)).
Проекцию Π_C и эталонное решение (SLSQP) берем из scripts/lab3_regularization_tradeoff.py.

Что получается:
  methods_n12.csv, methods_n50.csv      сравнение методов, один набор данных (seed=0)
  methods_seeds.csv                     то же, медиана и размах по 10 наборам данных
  start_point.csv                       проверка начальной точки
  unconstrained_vs_constrained.csv      решение без ограничений, после проекции и с ограничениями
  regularization.csv                    L2-регуляризация: исходная и изменённая задача
  summary_lab3.md                       все таблицы в markdown (для README)
  iterations_by_kappa.png               число итераций при n=12 и n=50: медиана по 10 наборам (с сортировкой)
  convergence_n12.png, convergence_n50.png
  regularization_tradeoff.png           итерации (столбики) и потеря на исходной функции (линия)
  regularization_distance.png           итерации (столбики) и сдвиг решения (линия)
  portfolio_weights.png                 как ограничения меняют состав портфеля
"""
from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oracles.constrained_markowitz import ConstrainedMarkowitzOracle
from scripts.lab3_regularization_tradeoff import (
    constrained_reference, feasible_projection,
)

# параметры
LAM = 0.05            # вес доходности (как в scripts/constraint_sensitivity.py)
CAP_SLOTS = 3         # w_max = 3/n: один актив не больше трёх «равных долей» (0.25 при n=12)
R_FRAC = 0.5          # r_min = r_base + R_FRAC * (r_max - r_base)
N_PERIODS = 300
DIMS = (12, 50)       # размеры задач: меньшая и большая обусловленность
SEED = 0              # основной набор данных
ROBUST_SEEDS = range(10)
TOL = 1e-8            # ||g_P|| <= TOL, g_P - отображение проецированного градиента
MAX_ITER = 20000
ALPHAS = [0, 1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3]
EPSILONS = [0.03, 0.10, 0.25]
SORT_DESCENDING = True   # порядок методов на графике: True - от большего числа итераций к меньшему

METHODS = [  # (ключ, название)
    ("pgd1", "PGD, шаг 1/L"),
    ("pgd2", "PGD, шаг 2/(L+μ)"),
    ("nesterov", "PGD + Nesterov"),
    ("bb1", "PGD + BB1"),
    ("bb2", "PGD + BB2"),
    ("armijo", "PGD + Armijo"),
]
NAMES = dict(METHODS)

OUT = Path(__file__).resolve().parents[1] / "results" / "lab3_final"


# задача
def build_oracle(dim, seed=SEED):
    """Оракул с ограничениями. r_min задаём выше доходности «базового» портфеля
    (бюджет + неотрицательность + w_max), чтобы ограничение по доходности было значимым."""
    cap = CAP_SLOTS / dim
    kw = dict(dim=dim, n_periods=N_PERIODS, seed=seed, lam=LAM, w_max=cap)
    base = ConstrainedMarkowitzOracle(**kw)
    r_base = base.ret(constrained_reference(base))
    r_max = base.max_feasible_return()
    r_min = r_base + R_FRAC * (r_max - r_base)
    return ConstrainedMarkowitzOracle(**kw, r_min=r_min)


def reference_solution(oracle, alpha=0.0):
    """Эталон для сравнения: грубое решение SLSQP, затем доводка проекционным методом
    до ||g_P|| <= 1e-12 (SLSQP останавливается по изменению функции и в «плоских»
    направлениях может отстоять от оптимума на ~1e-2 по весам). Возвращает (w, ||g_P||)."""
    w0 = feasible_projection(constrained_reference(oracle, alpha), oracle)
    for key, max_iter in (("bb1", 2000), ("nesterov", 200000)):
        r = run_method(oracle, key, w0, alpha=alpha, tol=1e-12, max_iter=max_iter)
        if r["converged"]:
            return r["x"], r["pg_norm"]
    raise RuntimeError("Не удалось получить точный эталон")


def kappa_of(oracle, alpha=0.0):
    ev = np.linalg.eigvalsh(oracle.A)
    return float((ev[-1] + alpha) / (ev[0] + alpha))


# методы
def run_method(oracle, key, x0, alpha=0.0, tol=TOL, max_iter=MAX_ITER):
    """Проекционный метод. Остановка у всех одинаковая: ||g_P(x)|| <= tol,
    g_P(x) = (x - Π_C(x - ∇F(x)/L)) * L, L - константа Липшица градиента F = f + α/2 ||w||²."""
    ev = np.linalg.eigvalsh(oracle.A)
    L, mu = ev[-1] + alpha, ev[0] + alpha
    eta_ref = 1.0 / L

    def grad(x):
        return oracle.grad(x) + alpha * x

    def F(x):
        return oracle.f(x) + 0.5 * alpha * float(x @ x)

    def P(v):
        return feasible_projection(v, oracle)

    n_proj = 0                    # проекции, которые выполняет сам метод (без проекций для остановки)

    def P_step(v):
        nonlocal n_proj
        n_proj += 1
        return P(v)

    x, x_prev = x0.copy(), x0.copy()
    eta = eta_ref
    g_prev = x_bb_prev = None
    f_hist = [oracle.f(x)]
    max_viol = oracle.violations(x)["max"]
    t_method = 0.0
    k = 0
    while True:
        g = grad(x)
        t1 = time.perf_counter()
        x_ref = P(x - eta_ref * g)                      # проекция для критерия остановки
        dt_ref = time.perf_counter() - t1
        pg = float(np.linalg.norm(x - x_ref) / eta_ref)
        if pg <= tol or k >= max_iter:
            break
        t1 = time.perf_counter()
        if key == "pgd1":                               # шаг 1/L совпадает с проекцией для остановки
            x_new = x_ref
            n_proj += 1
            t_method += dt_ref
        elif key == "pgd2":
            x_new = P_step(x - 2.0 / (L + mu) * g)
        elif key == "nesterov":
            y = x + k / (k + 3.0) * (x - x_prev)
            x_new = P_step(y - eta_ref * grad(y))
            x_prev = x
        elif key in ("bb1", "bb2"):
            if g_prev is not None:
                s, d = x - x_bb_prev, g - g_prev
                sd = float(s @ d)
                if sd > 1e-30:
                    eta = float(s @ s) / sd if key == "bb1" else sd / float(d @ d)
            x_bb_prev, g_prev = x.copy(), g.copy()
            x_new = P_step(x - eta * g)
        elif key == "armijo":
            e, fx = eta * 2.0, F(x)
            for _ in range(50):
                x_new = P_step(x - e * g)
                if F(x_new) <= fx + 1e-4 * float(g @ (x_new - x)):
                    break
                e *= 0.5
            eta = e
        else:
            raise ValueError(key)
        if key != "pgd1":
            t_method += time.perf_counter() - t1
        x = x_new
        k += 1
        f_hist.append(oracle.f(x))
        max_viol = max(max_viol, oracle.violations(x)["max"])
    return {"x": x, "iterations": k, "projections": n_proj, "pg_norm": pg, "converged": pg <= tol,
            "time_ms": 1e3 * t_method, "f_hist": np.array(f_hist),
            "max_violation_traj": float(max_viol)}


def unconstrained_gd_iterations(oracle, x0, tol=1e-8):
    """Для сравнения: обычный GD без ограничений (как в ЛР2) на тех же Σ, μ, λ."""
    L = np.linalg.eigvalsh(oracle.A)[-1]
    x, k = x0.copy(), 0
    while np.linalg.norm(oracle.grad(x)) > tol and k < 500000:
        x = x - oracle.grad(x) / L
        k += 1
    return k


# таблицы
def pretty(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return "да" if v else "нет"
    if isinstance(v, (float, np.floating)):
        if v == 0:
            return "0"
        return f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) >= 1e5) else f"{v:.{digits}g}"
    return str(v)


def md_table(rows, cols, heads):
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join(["---"] * len(heads)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(pretty(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_csv(path, rows):
    names = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def solution_row(oracle, w, w_star, f_star, **extra):
    v = oracle.violations(w)
    return {**extra, "f_original": oracle.f(w), "gap_f": max(0.0, oracle.f(w) - f_star),
            "dist_to_ref": float(np.linalg.norm(w - w_star)),
            "max_violation": float(v["max"]), "feasible": bool(oracle.is_feasible(w, 1e-8))}


# эксперименты
def methods_experiment(dim, seed=SEED):
    oracle = build_oracle(dim, seed)
    w_star, ref_pg = reference_solution(oracle)
    f_star = oracle.f(w_star)
    x0 = feasible_projection(np.full(dim, 1.0 / dim), oracle)
    assert oracle.is_feasible(x0, 1e-8), "начальная точка недопустима"
    kappa = kappa_of(oracle)

    runs, rows = {}, []
    for key, name in METHODS:
        r = run_method(oracle, key, x0)
        runs[key] = r
        rows.append({"method": name, "kappa": kappa, "iterations": r["iterations"],
                     "projections": r["projections"], "time_ms": r["time_ms"], "pg_norm": r["pg_norm"],
                     "max_violation_traj": r["max_violation_traj"],
                     **solution_row(oracle, r["x"], w_star, f_star)})
    base_it = rows[0]["iterations"]
    for r in rows:
        r["speedup"] = base_it / max(r["iterations"], 1)
    return oracle, w_star, f_star, x0, runs, rows, ref_pg


def robustness_experiment():
    """Те же методы на 10 наборах данных: медиана и размах числа итераций."""
    rows = []
    for dim in DIMS:
        per_method = {k: [] for k, _ in METHODS}
        kappas, skipped = [], 0
        for seed in ROBUST_SEEDS:
            try:
                oracle = build_oracle(dim, seed)
                x0 = feasible_projection(np.full(dim, 1.0 / dim), oracle)
            except Exception:
                skipped += 1
                continue
            kappas.append(kappa_of(oracle))
            for key, _ in METHODS:
                r = run_method(oracle, key, x0)
                per_method[key].append(r["iterations"] if r["converged"] else np.nan)
        for key, name in METHODS:
            v = np.array(per_method[key], dtype=float)
            rows.append({"n": dim, "key": key, "method": name, "median_kappa": float(np.median(kappas)),
                         "n_datasets": len(v), "iterations_median": float(np.nanmedian(v)),
                         "iterations_min": float(np.nanmin(v)), "iterations_max": float(np.nanmax(v))})
    return rows


def start_point_experiment(oracle, w_star, f_star):
    """Допустимость и «хорошесть» начальной точки + влияние на число итераций."""
    rng = np.random.default_rng(1)
    starts = {
        "Равные веса 1/n, спроецированные на C": feasible_projection(np.full(oracle.dim, 1.0 / oracle.dim), oracle),
        "Внутренняя точка (oracle.feasible_start)": oracle.feasible_start(),
        "Случайный вектор, спроецированный на C": feasible_projection(rng.normal(0.0, 1.0, oracle.dim), oracle),
    }
    L = np.linalg.eigvalsh(oracle.A)[-1]
    rows = []
    for name, x0 in starts.items():
        if x0 is None or not oracle.is_feasible(x0, 1e-8):
            rows.append({"start": name, "feasible": False})
            continue
        g_p = np.linalg.norm(x0 - feasible_projection(x0 - oracle.grad(x0) / L, oracle)) * L
        rows.append({"start": name, "feasible": True,
                     "max_violation": oracle.violations(x0)["max"],
                     "f_gap_at_start": oracle.f(x0) - f_star,
                     "dist_to_ref": float(np.linalg.norm(x0 - w_star)),
                     "pg_norm_at_start": float(g_p),
                     "iterations_pgd1": run_method(oracle, "pgd1", x0)["iterations"],
                     "iterations_bb1": run_method(oracle, "bb1", x0)["iterations"]})
    return rows


def projection_experiment(oracle, w_star, f_star):
    """Решение без ограничений (ЛР2), оно же после проекции и решение с ограничениями."""
    w_unc = np.linalg.solve(oracle.A, oracle.b_lam)
    w_proj = feasible_projection(w_unc, oracle)
    rows = []
    for name, w in [("Оптимум без ограничений (ЛР2)", w_unc),
                    ("Он же после проекции на C", w_proj),
                    ("Оптимум с ограничениями (эталон)", w_star)]:
        v = oracle.violations(w)
        rows.append({"portfolio": name, "sum_w": float(w.sum()), "min_w": float(w.min()),
                     "max_w": float(w.max()), "expected_return": oracle.ret(w),
                     "variance": oracle.risk(w), "f_original": oracle.f(w),
                     "gap_f": oracle.f(w) - f_star,
                     "dist_to_ref": float(np.linalg.norm(w - w_star)),
                     "viol_sum": float(v["sum"]), "viol_nonneg": float(v["nonneg"]),
                     "viol_cap": float(v["cap"]), "viol_return": float(v["return"]),
                     "feasible": bool(oracle.is_feasible(w, 1e-8))})
    return rows, (w_unc, w_proj)


def regularization_experiment(oracle, w_star, f_star, x0):
    rows = []
    for alpha in ALPHAS:
        r = run_method(oracle, "pgd1", x0, alpha=alpha)
        w = r["x"]
        w_ref, _ = reference_solution(oracle, alpha)
        F = lambda z: oracle.f(z) + 0.5 * alpha * float(z @ z)
        v = oracle.violations(w)
        dist = float(np.linalg.norm(w - w_star))
        row = {"alpha": alpha, "kappa": kappa_of(oracle, alpha), "iterations": r["iterations"],
               "converged": r["converged"],
               "f_original": oracle.f(w), "F_modified": F(w),
               "loss_original": max(0.0, oracle.f(w) - f_star),
               "loss_share_percent": 100 * max(0.0, oracle.f(w) - f_star) / (oracle.f(x0) - f_star),
               "gap_modified": max(0.0, F(w) - F(w_ref)),
               "dist_to_original_optimum": dist,
               "dist_to_regularized_optimum": float(np.linalg.norm(w - w_ref)),
               "max_violation": float(v["max"]),
               "expected_return": oracle.ret(w), "variance": oracle.risk(w)}
        for eps in EPSILONS:
            row[f"within_eps_{eps:g}"] = dist <= eps
        rows.append(row)
    return rows


# графики
def plot_iterations_by_kappa(rob_rows, path):
    """Число итераций при двух размерах задачи (медиана по наборам данных, усы - мин и макс).
    Методы упорядочены по числу итераций на большой задаче: SORT_DESCENDING=True - по убыванию."""
    by = {(r["n"], r["key"]): r for r in rob_rows}
    keys = [k for k, _ in METHODS]
    big = DIMS[-1]
    keys.sort(key=lambda k: by[(big, k)]["iterations_median"], reverse=SORT_DESCENDING)
    labels = [NAMES[k].replace(", шаг", "\nшаг").replace(" + ", "\n+ ") for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#F58518", "#4C78A8"]
    top = 0
    for j, dim in enumerate(DIMS):
        med = np.array([by[(dim, k)]["iterations_median"] for k in keys])
        lo = np.array([by[(dim, k)]["iterations_min"] for k in keys])
        hi = np.array([by[(dim, k)]["iterations_max"] for k in keys])
        kappa = by[(dim, keys[0])]["median_kappa"]
        bars = ax.bar(x + (j - 0.5) * w, med, w, color=colors[j], label=f"n = {dim},  κ ≈ {kappa:.0f}",
                      yerr=[med - lo, hi - med], error_kw={"elinewidth": 1, "capsize": 3, "alpha": 0.7})
        ax.bar_label(bars, labels=[f"{v:g}" for v in med], padding=3, fontsize=9)
        top = max(top, hi.max())
    ax.set_yscale("log")
    ax.set_ylim(0.8, top * 4)
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"число итераций до ‖g_P‖ ≤ {TOL:g} (log)")
    ax.set_title(f"Число итераций проекционных методов: медиана по {len(ROBUST_SEEDS)} наборам данных")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_convergence(runs, f_star, dim, kappa, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, name in METHODS:
        gap = np.maximum(runs[key]["f_hist"] - f_star, 1e-18)
        ax.semilogy(gap, marker="o" if len(gap) < 15 else None, ms=4, label=name)
    ax.set_xscale("symlog")
    ax.set_xlabel("итерация (symlog)")
    ax.set_ylabel("f(w_k) − f(w*)  (исходная функция)")
    ax.set_title(f"Сходимость проекционных методов, n = {dim}, κ ≈ {kappa:.0f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_regularization(rows, line_key, line_label, title, path, eps_lines=False, percent=False):
    """Синие столбики - число итераций, красная линия с точками - цена регуляризации."""
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(rows))
    labels = ["0" if r["alpha"] == 0 else f"{r['alpha']:g}" for r in rows]
    bars = ax1.bar(x, [r["iterations"] for r in rows], 0.55, color="#4C78A8")
    ax1.bar_label(bars, padding=2, fontsize=9, color="#4C78A8")
    ax1.set_xticks(x, labels)
    ax1.set_xlabel("коэффициент регуляризации α")
    ax1.set_ylabel("итерации PGD", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(x, [r[line_key] for r in rows], "o-", color="#E45756")
    ax2.set_ylabel(line_label, color="#E45756")
    if eps_lines:
        for eps in EPSILONS:
            ax2.axhline(eps, ls="--", lw=1, color="gray", alpha=0.7)
            ax2.text(len(rows) - 0.55, eps, f"ε = {eps:g}", va="bottom", ha="right", fontsize=8, color="gray")
    ax1.set_title(title)
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_weights(w_unc, w_proj, w_star, cap, path):
    n = len(w_star)
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w, w_unc, w, label=f"без ограничений (ЛР2), Σw = {w_unc.sum():.2f}", color="#9D9D9D")
    ax.bar(x, w_proj, w, label="после проекции на C", color="#F58518")
    ax.bar(x + w, w_star, w, label="оптимум с ограничениями", color="#4C78A8")
    ax.axhline(cap, ls="--", color="k", lw=1, label=f"w_max = {cap:.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{i + 1}" for i in x])
    ax.set_xlabel("актив")
    ax.set_ylabel("доля капитала")
    ax.set_title("Как ограничения меняют состав портфеля")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_dim, kappas, main_ctx = {}, {}, {}
    md = []

    for dim in DIMS:
        oracle, w_star, f_star, x0, runs, rows, ref_pg = methods_experiment(dim)
        rows_by_dim[dim] = rows
        kappas[dim] = rows[0]["kappa"]
        main_ctx[dim] = (oracle, w_star, f_star, x0, runs)
        unc_it = unconstrained_gd_iterations(oracle, x0)
        write_csv(OUT / f"methods_n{dim}.csv", rows)
        plot_convergence(runs, f_star, dim, kappas[dim], OUT / f"convergence_n{dim}.png")
        print(f"\n=== n={dim}: kappa={kappas[dim]:.1f}, w_max={oracle.w_max:.4g}, "
              f"r_min={oracle.r_min:.6g}, GD без ограничений: {unc_it} итераций, "
              f"невязка эталона ||g_P(w*)||={ref_pg:.1e} ===")
        for r in rows:
            print(f"{r['method']:<20} it={r['iterations']:>5} proj={r['projections']:>5} gap={r['gap_f']:.1e} "
                  f"dist={r['dist_to_ref']:.1e} viol={r['max_violation']:.1e}")
        md += [f"### Задача n = {dim} (κ ≈ {kappas[dim]:.0f}, w_max = {oracle.w_max:.3g}); "
               f"GD без ограничений на тех же данных: {unc_it} итераций\n",
               md_table(rows, ["method", "iterations", "projections", "speedup", "gap_f", "dist_to_ref",
                               "max_violation_traj", "feasible"],
                        ["Метод", "Итерации", "Проекций на C", "Ускорение к PGD 1/L", "f − f* (исходная)",
                         "‖w − w*‖", "Макс. нарушение", "Допустим"]), ""]

    oracle, w_star, f_star, x0, _ = main_ctx[DIMS[0]]
    rob = robustness_experiment()
    write_csv(OUT / "methods_seeds.csv", rob)
    plot_iterations_by_kappa(rob, OUT / "iterations_by_kappa.png")
    md += ["### Устойчивость: медиана (мин–макс) числа итераций по 10 наборам данных\n",
           md_table([{**r, "range": f"{r['iterations_min']:.0f}–{r['iterations_max']:.0f}"} for r in rob],
                    ["n", "method", "median_kappa", "iterations_median", "range"],
                    ["n", "Метод", "κ (медиана)", "Итерации (медиана)", "Размах"]), ""]

    start_rows = start_point_experiment(oracle, w_star, f_star)
    write_csv(OUT / "start_point.csv", start_rows)
    md += [f"### Начальная точка (n = {DIMS[0]})\n",
           md_table(start_rows, ["start", "feasible", "f_gap_at_start", "dist_to_ref", "pg_norm_at_start",
                                 "iterations_pgd1", "iterations_bb1"],
                    ["Начальная точка", "Допустима", "f(w₀) − f*", "‖w₀ − w*‖", "‖g_P(w₀)‖",
                     "Итер. PGD 1/L", "Итер. BB1"]), ""]

    proj_rows, (w_unc, w_proj) = projection_experiment(oracle, w_star, f_star)
    write_csv(OUT / "unconstrained_vs_constrained.csv", proj_rows)
    plot_weights(w_unc, w_proj, w_star, oracle.w_max, OUT / "portfolio_weights.png")
    md += [f"### Решение без ограничений, после проекции и с ограничениями (n = {DIMS[0]})\n",
           md_table(proj_rows, ["portfolio", "sum_w", "min_w", "max_w", "expected_return", "f_original",
                                "gap_f", "dist_to_ref", "feasible"],
                    ["Портфель", "Σw", "min w", "max w", "Ожид. доходность", "f (исходная)",
                     "f − f*", "‖w − w*‖", "Допустим"]), ""]

    reg_rows = regularization_experiment(oracle, w_star, f_star, x0)
    write_csv(OUT / "regularization.csv", reg_rows)
    plot_regularization(reg_rows, "loss_share_percent", "потеря на исходной f, % от достигнутого улучшения",
                        "L2-регуляризация: число итераций и потеря на исходной функции",
                        OUT / "regularization_tradeoff.png")
    plot_regularization(reg_rows, "dist_to_original_optimum", "‖w_α − w*‖  (сдвиг решения)",
                        "L2-регуляризация: число итераций и сдвиг решения от исходного",
                        OUT / "regularization_distance.png", eps_lines=True)
    md += [f"### L2-регуляризация (n = {DIMS[0]})\n",
           md_table(reg_rows, ["alpha", "kappa", "iterations", "f_original", "F_modified",
                               "loss_share_percent", "dist_to_original_optimum", "max_violation"],
                    ["α", "κ(α)", "Итерации", "f (исходная)", "F_α (изменённая)", "Потеря, % от улучшения",
                     "‖w_α − w*‖", "Макс. нарушение"]), ""]

    (OUT / "summary_lab3.md").write_text("\n".join(md), encoding="utf-8")
    print("\nНачальная точка:")
    for r in start_rows:
        print("  ", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nL2-регуляризация:")
    for r in reg_rows:
        print(f"  alpha={r['alpha']:g} kappa={r['kappa']:.2f} it={r['iterations']} "
              f"loss={r['loss_share_percent']:.2f}% dist={r['dist_to_original_optimum']:.3f}")
    print("\nФайлы сохранены в:", OUT)


if __name__ == "__main__":
    main()
