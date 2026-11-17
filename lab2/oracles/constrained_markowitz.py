import numpy as np
from scipy.optimize import linprog, minimize
from .quadratic import QuadraticOracle

class ConstrainedMarkowitzOracle(QuadraticOracle):
    """Марковиц с ограничениями (ЛР3).

    min f(w) = 1/2 w^T S w - lam * m^T w
    при: 1^T w = 1        (весь капитал вложен)
         w_i >= 0         (без коротких позиций)
         w_i <= w_max     (не более w_max капитала в один актив)
         m^T w >= r_min   (минимально допустимая ожидаемая доходность)
         w^T S w <= risk_max (опционально: максимальная дисперсия)
    Здесь S = self.A (оценка ковариации), m = self.b (средние доходности).
    """

    def __init__(self, dim=20, n_periods=300, seed=0, lam=1.0,
                 w_max=0.3, r_min=None, risk_max=None):
        super().__init__(dim=dim, n_periods=n_periods, seed=seed)
        self.lam = lam
        self.w_max = w_max
        self.r_min = r_min
        self.risk_max = risk_max
        self.name = (f"markowitz-constrained(dim={dim}, w_max={w_max}, "
                     f"r_min={r_min}, risk_max={risk_max})")
        # f, grad, hess с учётом lam
        self.b_lam = lam * self.b

    # --- целевая функция ---
    def f(self, w): return 0.5 * w @ self.A @ w - self.b_lam @ w
    def grad(self, w): return self.A @ w - self.b_lam
    def hess(self, w): return self.A

    # --- ограничения ---
    def risk(self, w): return float(w @ self.A @ w)
    def ret(self, w): return float(self.b @ w)

    def violations(self, w):
        """Нарушение каждого ограничения (0 = выполнено)."""
        v = {
            "sum": abs(w.sum() - 1.0),
            "nonneg": max(0.0, -w.min()),
            "cap": max(0.0, w.max() - self.w_max),
            "return": 0.0 if self.r_min is None else max(0.0, self.r_min - self.ret(w)),
            "risk": 0.0 if self.risk_max is None else max(0.0, self.risk(w) - self.risk_max),
        }
        v["max"] = max(v.values())
        return v

    def is_feasible(self, w, tol=1e-9):
        return self.violations(w)["max"] <= tol

    # --- характеристики допустимой области ---
    def max_feasible_return(self):
        """Максимум m^T w при sum=1, 0<=w<=w_max (LP)."""
        res = linprog(-self.b, A_eq=np.ones((1, self.dim)), b_eq=[1],
                      bounds=[(0, self.w_max)] * self.dim)
        return -res.fun if res.success else None

    def min_feasible_risk(self):
        """Минимум w^T S w при sum=1, 0<=w<=w_max (без r_min)."""
        w0 = np.full(self.dim, 1 / self.dim)
        res = minimize(self.risk, w0, jac=lambda w: 2 * self.A @ w, method="SLSQP",
                       bounds=[(0, self.w_max)] * self.dim,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        return res.fun

    def feasible_region_nonempty(self):
        if self.dim * self.w_max < 1 - 1e-12:
            return False
        if self.r_min is not None and self.r_min > self.max_feasible_return() + 1e-12:
            return False
        return self.solve_reference(check_only=True) is not None

    # --- допустимая начальная точка ---
    def feasible_start(self):
        """1) равные веса 1/n; 2) если не допустимы - LP с максимальным запасом
        до границ и по доходности; 3) если нарушен риск - SLSQP по минимизации риска."""
        n = self.dim
        w = np.full(n, 1.0 / n)
        if self.is_feasible(w):
            return w
        # LP: max t,  t <= w_i <= w_max - t,  m^T w >= r_min + t*eps_r
        c = np.zeros(n + 1); c[-1] = -1
        A_ub, b_ub = [], []
        for i in range(n):
            row = np.zeros(n + 1); row[i] = -1; row[-1] = 1; A_ub.append(row); b_ub.append(0)
            row = np.zeros(n + 1); row[i] = 1; row[-1] = 1; A_ub.append(row); b_ub.append(self.w_max)
        if self.r_min is not None:
            row = np.zeros(n + 1); row[:n] = -self.b; row[-1] = 1e-4
            A_ub.append(row); b_ub.append(-self.r_min)
        A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1
        res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=[1],
                      bounds=[(None, None)] * n + [(0, None)])
        if not res.success:
            return None
        w = res.x[:n]
        if self.is_feasible(w, 1e-8):
            return w
        ref = self.solve_reference(check_only=True)
        return ref

    # --- эталонное решение (SLSQP) ---
    def solve_reference(self, check_only=False):
        n = self.dim
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
        if self.r_min is not None:
            cons.append({"type": "ineq", "fun": lambda w: self.ret(w) - self.r_min,
                         "jac": lambda w: self.b})
        if self.risk_max is not None:
            cons.append({"type": "ineq", "fun": lambda w: self.risk_max - self.risk(w),
                         "jac": lambda w: -2 * self.A @ w})
        w0 = np.full(n, 1.0 / n)
        res = minimize(self.f, w0, jac=self.grad, method="SLSQP",
                       bounds=[(0, self.w_max)] * n, constraints=cons,
                       options={"ftol": 1e-14, "maxiter": 1000})
        if not res.success or not self.is_feasible(res.x, 1e-7):
            return None
        return res.x if check_only else (res.x, res.fun)
