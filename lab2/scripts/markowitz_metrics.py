import numpy as np

from oracles import QuadraticOracle
from methods import GD
from utils.metrics import Timer, summarize


oracle = QuadraticOracle(dim=2, n_periods=300, seed=0)

method = GD(
    oracle,
    max_iter=5000,
    tol=1e-8
)

x0 = np.zeros(oracle.dim)

with Timer() as timer:
    history = method.run(x0)

metrics = summarize(
    history,
    oracle=oracle,
    tol=1e-8
)

metrics["time_sec"] = timer.elapsed

print("=== Markowitz GD metrics ===")
for key, value in metrics.items():
    print(f"{key}: {value}")
