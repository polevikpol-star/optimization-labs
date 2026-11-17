
import numpy as np
from scipy.optimize import minimize


def project_capped_simplex(v, w_max, tol=1e-12):
    """Точная с численной погрешностью проекция:
    sum(w)=1, 0<=w_i<=w_max.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)

    if n * w_max < 1 - tol:
        raise ValueError("Допустимая область пуста")

    low = np.min(v) - w_max
    high = np.max(v)

    for _ in range(150):
        theta = (low + high) / 2
        w = np.clip(v - theta, 0.0, w_max)

        if np.sum(w) > 1:
            low = theta
        else:
            high = theta

    return np.clip(v - (low + high) / 2, 0.0, w_max)


def project_portfolio(v, oracle):
    """Проекция с учётом всех включённых ограничений,
    кроме risk_max.
    """
    if oracle.risk_max is not None:
        raise NotImplementedError(
            "Ограничение риска пока не поддерживается"
        )

    v = np.asarray(v, dtype=float)

    # Без ограничения доходности используем
    # быструю проекцию на ограниченный симплекс.
    if oracle.r_min is None:
        w = project_capped_simplex(v, oracle.w_max)

        if not oracle.is_feasible(w, tol=1e-7):
            raise RuntimeError("Ошибка проекции")

        return w

    # С минимальной доходностью решаем
    # квадратичную задачу через SLSQP.
    n = oracle.dim

    x0 = oracle.feasible_start()
    if x0 is None:
        raise ValueError("Нет допустимой начальной точки")

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
            "jac": lambda w: np.ones(n),
        },
        {
            "type": "ineq",
            "fun": lambda w: oracle.b @ w - oracle.r_min,
            "jac": lambda w: oracle.b,
        },
    ]

    result = minimize(
        lambda w: 0.5 * np.sum((w - v) ** 2),
        x0=x0,
        jac=lambda w: w - v,
        method="SLSQP",
        bounds=[(0, oracle.w_max)] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(
            f"Проекция не сошлась: {result.message}"
        )

    if not oracle.is_feasible(result.x, tol=1e-7):
        raise RuntimeError("Недопустимая проекция")

    return result.x
