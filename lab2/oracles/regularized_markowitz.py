"""L2-regularization of an existing quadratic Markowitz oracle."""
import numpy as np
from .base import Oracle, OracleInfo


class RegularizedMarkowitzOracle(Oracle):
    """f_alpha(w) = f(w) + alpha/2 * ||w||^2, alpha >= 0."""

    def __init__(self, base_oracle, alpha: float):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        self.base = base_oracle
        self.alpha = float(alpha)
        self.dim = base_oracle.dim
        self.name = f'{base_oracle.name}_l2(alpha={alpha:g})'
        self.A = base_oracle.A + self.alpha * np.eye(self.dim)
        self.b = base_oracle.b.copy()
        eig = np.linalg.eigvalsh(self.A)
        if eig[0] <= 0:
            raise ValueError('regularized Hessian must be positive definite')
        self._info = OracleInfo(
            L=float(eig[-1]), mu=float(eig[0]),
            x_star=np.linalg.solve(self.A, self.b),
        )
        x = self._info.x_star
        self._info.f_star = self.f(x)

    def f(self, x):
        return float(self.base.f(x) + 0.5 * self.alpha * np.dot(x, x))

    def grad(self, x):
        return self.base.grad(x) + self.alpha * x

    def hess(self, x):
        return self.A.copy()

    @property
    def info(self):
        return self._info
