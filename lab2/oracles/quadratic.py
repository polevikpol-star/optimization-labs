import numpy as np
from .base import Oracle, OracleInfo


class QuadraticOracle(Oracle):
    """f(x) = 1/2 x^T A x - b^T x.

    A — симметричная положительно определённая с заданным числом
    обусловленности kappa = L / mu.
    """

    def __init__(self, dim: int = 2, n_periods: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        
        mu_true = rng.uniform(0.0005, 0.002, dim)
        beta = rng.uniform(0.8, 1.5, dim)
        
        z = rng.normal(0, 0.01, n_periods)
        epsilon = rng.normal(0, 0.005, (n_periods, dim))
        
        r = mu_true + np.outer(z, beta) + epsilon
        
        self.b = np.mean(r, axis=0)
        cov = np.cov(r, rowvar=False)
        self.A = cov + 1e-6 * np.eye(dim)
        
        self.dim = dim
        self.name = f"markowitz(dim={dim}, periods={n_periods})"
        
        eigs = np.linalg.eigvalsh(self.A)
        self._L = eigs.max()
        self._mu = eigs.min()
        self._x_star = np.linalg.solve(self.A, self.b)
        self._f_star = 0.5 * self._x_star @ self.A @ self._x_star - self.b @ self._x_star

    def f(self, x: np.ndarray) -> float:
        return 0.5 * x @ self.A @ x - self.b @ x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self.A

    @property
    def info(self) -> OracleInfo:
        return OracleInfo(L=self._L, mu=self._mu,
                          x_star=self._x_star, f_star=self._f_star)
