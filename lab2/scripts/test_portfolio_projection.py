
import numpy as np

from oracles.constrained_markowitz import ConstrainedMarkowitzOracle
from utils.portfolio_projection import project_portfolio


oracle = ConstrainedMarkowitzOracle(
    dim=20,
    n_periods=300,
    seed=0,
    w_max=0.3,
    r_min=None,
    risk_max=None,
)

# Начальная точка должна быть допустимой
x0 = oracle.feasible_start()

print("НАЧАЛЬНАЯ ТОЧКА")
print("Найдена:", x0 is not None)

if x0 is None:
    raise RuntimeError("Нет допустимой начальной точки")

print("Допустима:", oracle.is_feasible(x0))
print("Сумма весов:", np.sum(x0))
print("Максимальный вес:", np.max(x0))
print("Нарушения:", oracle.violations(x0))

# Создаём заведомо недопустимый портфель
v = np.zeros(oracle.dim)
v[0] = 0.8
v[1] = 0.2

print("\nДО ПРОЕКЦИИ")
print("Сумма весов:", np.sum(v))
print("Максимальный вес:", np.max(v))
print("Нарушения:", oracle.violations(v))

w = project_portfolio(v, oracle)

print("\nПОСЛЕ ПРОЕКЦИИ")
print("Сумма весов:", np.sum(w))
print("Минимальный вес:", np.min(w))
print("Максимальный вес:", np.max(w))
print("Нарушения:", oracle.violations(w))
print("Допустима:", oracle.is_feasible(w, tol=1e-7))

assert oracle.is_feasible(w, tol=1e-7)
print("\nТЕСТ ПРОЙДЕН")
