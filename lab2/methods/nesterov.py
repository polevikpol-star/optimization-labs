import numpy as np
from .base import Method


class Nesterov(Method):
    """Ускоренный градиентный метод Нестерова.

    y_k = x_k + beta_k (x_k - x_{k-1});  x_{k+1} = y_k - eta * grad f(y_k)
    """

    def __init__(self, oracle, step=None, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        if step is None:
            L = oracle.info.L
            step = 1.0 / L if L else 1e-3
        self.step = float(step)
        self.name = "nesterov"

    def _step(self, state):
        x, x_prev = state["x"], state["x_prev"]
        k = state["k"]                      # номер уже сделанных итераций
        beta = k / (k + 3.0)
        y = x + beta * (x - x_prev)
        g = self.oracle.grad(y)
        state["x_prev"] = x
        state["x"] = y - self.step * g
        state["step"] = self.step
        return state