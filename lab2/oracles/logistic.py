import numpy as np
from .base import Oracle, OracleInfo


class LogisticOracle(Oracle):
    """L2-регуляризованная логистическая регрессия.

    f(w) = 1/n sum log(1 + exp(-y_i * w^T x_i)) + l2/2 * ||w||^2
    """

    def __init__(self, n: int = 1000, dim: int = 50,
                 l2: float = 1.0, seed: int = 0):
        if n <= 0 or dim <= 0:
            raise ValueError("n and dim must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n, dim))
        self.y = np.sign(rng.standard_normal(n))
        self.y[self.y == 0] = 1.0
        self.l2 = l2
        self.dim = dim
        self.name = f"logistic(n={n}, dim={dim}, l2={l2:g})"

        # грубая оценка гладкости: ||X||^2 / (4n) + l2
        sigma_max = np.linalg.norm(self.X, ord=2)
        self._L = sigma_max ** 2 / (4.0 * n) + l2
        self._mu = l2  # нижняя граница

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid evaluated elementwise."""
        result = np.empty_like(z, dtype=float)
        positive = z >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)
        return result

    def f(self, w: np.ndarray) -> float:
        z = -self.y * (self.X @ w)
        loss = np.logaddexp(0.0, z).mean()
        return float(loss + 0.5 * self.l2 * w @ w)

    def grad(self, w: np.ndarray) -> np.ndarray:
        z = -self.y * (self.X @ w)
        sig = self._sigmoid(z)
        g = self.X.T @ (-self.y * sig) / len(self.y)
        return g + self.l2 * w
    
    def hess(self, w: np.ndarray) -> np.ndarray:
        z = -self.y * (self.X @ w)
        sig = self._sigmoid(z)
        curvature = sig * (1.0 - sig)
        return self.X.T @ (self.X * curvature[:, None]) / len(self.y) + (
            self.l2 * np.eye(self.dim)
        )

    @property
    def info(self) -> OracleInfo:
        return OracleInfo(L=self._L, mu=self._mu)
