"""ЛР3: влияние ограничений на решение модели Марковица.

Поместить в lab2/scripts/constraint_sensitivity.py.
Запуск из lab2/: python -m scripts.constraint_sensitivity

Используются данные и формулы командного ConstrainedMarkowitzOracle.
Эталонные решения вычисляются SLSQP, а не Projected GD.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog, minimize
from oracles.constrained_markowitz import ConstrainedMarkowitzOracle

DIM = 12
PERIODS = 300
SEED = 0
LAM = 0.05  # Фиксируем для всех опытов: видна чувствительность доходности к ограничениям
BASE_CAP = 0.25
OUT = Path(__file__).resolve().parents[1] / "results" / "constraint_sensitivity"


def make_oracle(cap=BASE_CAP, r_min=None, risk_max=None):
    return ConstrainedMarkowitzOracle(dim=DIM, n_periods=PERIODS, seed=SEED,
                                      lam=LAM, w_max=cap, r_min=r_min, risk_max=risk_max)


def initial_feasible_point(oracle):
    """Ищем допустимую точку линейным программированием (без ограничения риска)."""
    cons = []
    if oracle.r_min is not None:
        cons.append((-oracle.b.reshape(1, -1), np.array([-oracle.r_min])))
    A_ub = np.vstack([p[0] for p in cons]) if cons else None
    b_ub = np.concatenate([p[1] for p in cons]) if cons else None
    res = linprog(np.zeros(oracle.dim), A_ub=A_ub, b_ub=b_ub,
                  A_eq=np.ones((1, oracle.dim)), b_eq=[1.0],
                  bounds=[(0, oracle.w_max)] * oracle.dim,
                  method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                            "dual_feasibility_tolerance": 1e-9})
    return res.x if res.success else None


def solve_portfolio(oracle, objective="markowitz", start=None):
    """Масштабируем целевую функцию и ограничения для устойчивого SLSQP.

    Возвращает (веса, результат solver) или (None, результат) при неудаче.
    """
    if oracle.dim * oracle.w_max < 1 - 1e-12:
        return None, None
    w0 = initial_feasible_point(oracle)
    if w0 is None:
        return None, None
    if start is not None and oracle.is_feasible(np.asarray(start), tol=1e-9):
        w0 = np.asarray(start, dtype=float)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1,
                    "jac": lambda w: np.ones(oracle.dim)}]
    if oracle.r_min is not None:
        constraints.append({"type": "ineq", "fun": lambda w: 1e4 * (oracle.ret(w) - oracle.r_min),
                            "jac": lambda w: 1e4 * oracle.b})
    if oracle.risk_max is not None:
        constraints.append({"type": "ineq", "fun": lambda w: 1e6 * (oracle.risk_max - oracle.risk(w)),
                            "jac": lambda w: -2e6 * oracle.A @ w})
    if objective == "markowitz":
        fun = lambda w: 1e4 * oracle.f(w)
        jac = lambda w: 1e4 * oracle.grad(w)
    elif objective == "min_risk":
        fun = lambda w: 1e6 * oracle.risk(w)
        jac = lambda w: 2e6 * oracle.A @ w
    else:
        raise ValueError("Unknown objective")
    starts = [w0, np.full(oracle.dim, 1 / oracle.dim)]
    if oracle.risk_max is not None:
        # Найдем допустимый старт по риску через задачу минимизации дисперсии.
        oracle_without_risk = make_oracle(cap=oracle.w_max, r_min=oracle.r_min)
        w_safe, _ = solve_portfolio(oracle_without_risk, objective="min_risk")
        if w_safe is not None:
            starts.insert(0, w_safe)
    best = None
    for candidate in starts:
        if not oracle.is_feasible(candidate, tol=1e-9):
            continue
        res = minimize(fun, candidate, jac=jac, method="SLSQP",
                       bounds=[(0, oracle.w_max)] * oracle.dim,
                       constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 3000})
        violation = oracle.violations(res.x)
        if violation["sum"] <= 1e-7 and violation["nonneg"] <= 1e-8 and violation["cap"] <= 1e-8 and violation["return"] <= 1e-9 and violation["risk"] <= 1e-9:
            if best is None or fun(res.x) < fun(best.x):
                best = res
    if best is None:
        return None, None
    return best.x, best


def max_return(oracle):
    res = linprog(-oracle.b, A_eq=np.ones((1, oracle.dim)), b_eq=[1],
                  bounds=[(0, oracle.w_max)] * oracle.dim, method="highs")
    return -float(res.fun) if res.success else None


def result_row(experiment, setting, oracle, reference_w, proof=None):
    """Сначала доказываем несовместность через границы, а не ошибку solver."""
    if proof is not None:
        return {"experiment": experiment, "setting": setting, "status": "infeasible_proven",
                "reason": proof}
    w, res = solve_portfolio(oracle)
    if w is None:
        return {"experiment": experiment, "setting": setting, "status": "solver_unconfirmed",
                "reason": "SLSQP did not return a certified-feasible solution; not proof of infeasibility"}
    v = oracle.violations(w)
    active = []
    if np.any(np.isclose(w, oracle.w_max, atol=1e-6)):
        active.append("max_weight")
    if oracle.r_min is not None and abs(oracle.ret(w) - oracle.r_min) < 1e-8:
        active.append("min_return")
    if oracle.risk_max is not None and abs(oracle.risk(w) - oracle.risk_max) < 1e-8:
        active.append("max_risk")
    return {"experiment": experiment, "setting": setting, "status": "feasible",
            "reason": "", "expected_return": oracle.ret(w), "portfolio_variance": oracle.risk(w),
            "portfolio_volatility_percent": 100 * np.sqrt(oracle.risk(w)),
            "original_objective": oracle.f(w),
            "distance_to_baseline": float(np.linalg.norm(w - reference_w)),
            "max_weight_used": float(w.max()), "active_constraints": ";".join(active),
            "max_constraint_violation": v["max"], "sum_violation": v["sum"],
            "cap_violation": v["cap"], "return_violation": v["return"],
            "risk_violation": v["risk"], "solver_success": res.success,
            "iterations_slsqp": res.nit, "weights": ";".join(f"{z:.9f}" for z in w)}


def write_csv(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def draw(experiment, rows, x_label, filename, reference_w, baseline_return):
    good = [r for r in rows if r["status"] == "feasible"]
    infeasible = [r for r in rows if r["status"] == "infeasible_proven"]
    raw_x = np.array([r["setting"] for r in good], dtype=float)
    transform = (lambda v: v*100) if experiment in ("max_weight", "min_return") else (lambda v: 100*np.sqrt(v))
    x = transform(raw_x)
    if experiment == "max_weight":
        x_label = "Максимальный вес, %"
    elif experiment == "min_return":
        x_label = "Минимальная доходность, % за период"
    else:
        x_label = "Максимально допустимое стандартное отклонение, %"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x, [100*r["expected_return"] for r in good], "o-")
    axes[0].axhline(100*baseline_return, linestyle="--", linewidth=1, label="Базовый портфель")
    axes[0].set_ylabel("Ожидаемая доходность, % за период")
    axes[1].plot(x, [r["portfolio_volatility_percent"] for r in good], "o-")
    axes[1].set_ylabel("Риск (стандартное отклонение), %")
    axes[2].plot(x, [r["distance_to_baseline"] for r in good], "o-")
    axes[2].set_ylabel("Расстояние до базового портфеля, L2")
    for ax in axes:
        for item in infeasible:
            ax.axvline(transform(float(item["setting"])), linestyle=":", alpha=0.6)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)
    if experiment == "max_weight":
        axes[0].legend(fontsize=8)
    fig.suptitle("Чувствительность оптимума: " + experiment)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)

    # Позиции всех активов: видно, как меняется сам портфель, не только доходность и риск.
    weights = np.array([[float(z) for z in r["weights"].split(";")] for r in good])
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    image = ax2.imshow(weights.T, aspect="auto", origin="lower", interpolation="nearest")
    ax2.set_yticks(np.arange(DIM), labels=["Актив {}".format(i+1) for i in range(DIM)])
    ax2.set_xticks(np.arange(len(good)), labels=["{:.5g}".format(t) for t in x], rotation=45, ha="right")
    ax2.set_xlabel(x_label)
    ax2.set_title("Оптимальные веса активов: " + experiment)
    fig2.colorbar(image, ax=ax2, label="Доля капитала")
    fig2.tight_layout()
    fig2.savefig(OUT / (experiment + "_weights.png"), dpi=180)
    plt.close(fig2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = make_oracle()
    w_base, res_base = solve_portfolio(base)
    if w_base is None:
        raise RuntimeError("Не найден допустимый базовый портфель")
    assert base.is_feasible(w_base, tol=1e-8)
    min_risk_oracle = make_oracle()
    w_min_risk, _ = solve_portfolio(min_risk_oracle, objective="min_risk")
    if w_min_risk is None:
        raise RuntimeError("Не удалось найти минимум риска")
    risk_min = base.risk(w_min_risk)
    risk_base = base.risk(w_base)
    return_base = base.ret(w_base)
    ret_max = max_return(base)
    if ret_max is None:
        raise RuntimeError("Не удалось найти максимальную доходность")

    rows = []
    # Опыт 1: ограничение на максимальную долю актива
    for cap in [0.075, 1/12, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70, 1.0]:
        oracle = make_oracle(cap=cap)
        proof = "12 * w_max < 1: сумма весов не может быть 1" if DIM*cap < 1-1e-12 else None
        rows.append(result_row("max_weight", cap, oracle, w_base, proof))

    # Опыт 2: минимальная ожидаемая доходность, при неизменном cap=0.25
    for fraction in [-0.50, 0, 0.5, 0.85, 0.95, 0.99, 1.0, 1.03]:
        rmin = return_base + fraction*(ret_max-return_base)
        oracle = make_oracle(r_min=rmin)
        proof = "r_min > max feasible return" if rmin > ret_max + 1e-12 else None
        rows.append(result_row("min_return", rmin, oracle, w_base, proof))

    # Опыт 3: ограничение на дисперсию, при неизменном cap=0.25
    gap = risk_base-risk_min
    for fraction in [-0.05, 0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25]:
        rmax = risk_min + fraction*gap
        oracle = make_oracle(risk_max=rmax)
        proof = "risk_max < minimum achievable variance" if rmax < risk_min - 1e-12 else None
        rows.append(result_row("max_risk", rmax, oracle, w_base, proof))

    write_csv(OUT / "all_results.csv", rows)
    for name, filename, label in [("max_weight", "max_weight.png", "Максимальный вес"),
                                  ("min_return", "min_return.png", "Минимальная доходность"),
                                  ("max_risk", "max_risk.png", "Максимальная дисперсия")]:
        subset = [r for r in rows if r["experiment"] == name]
        write_csv(OUT / (name + ".csv"), subset)
        draw(name, subset, label, filename, w_base, return_base)

    print("DATA: dim={}, periods={}, seed={}, lambda={}".format(DIM, PERIODS, SEED, LAM))
    print("Baseline: return={:.10g}, variance={:.10g}, volatility={:.4f}%, f={:.10g}".format(
        return_base, risk_base, 100*np.sqrt(risk_base), base.f(w_base)))
    print("Maximum feasible return={:.10g}, minimum feasible variance={:.10g}".format(ret_max, risk_min))
    for name in ["max_weight", "min_return", "max_risk"]:
        print("\nEXPERIMENT:", name)
        for row in rows:
            if row["experiment"] == name:
                print("setting={:.9g} status={} ret={} risk={} dist={} violation={}".format(
                    row["setting"], row["status"], round(row["expected_return"], 9) if "expected_return" in row else "-",
                    round(row["portfolio_variance"], 9) if "portfolio_variance" in row else "-",
                    round(row["distance_to_baseline"], 6) if "distance_to_baseline" in row else "-",
                    row.get("max_constraint_violation", "-")))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
