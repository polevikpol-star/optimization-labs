"""Run from lab2/: python -m scripts.compare_regularization"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oracles.quadratic import QuadraticOracle
from oracles.regularized_markowitz import RegularizedMarkowitzOracle
from methods.gd import GD


def main():
    out = Path(__file__).resolve().parents[1] / 'results' / 'regularization'
    out.mkdir(parents=True, exist_ok=True)
    base = QuadraticOracle(dim=2, n_periods=300, seed=0)
    x0 = np.zeros(base.dim)
    alphas = [0, 1e-6, 1e-5, 1e-4, 1e-3]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in alphas:
        oracle = RegularizedMarkowitzOracle(base, alpha)
        # Same optimal fixed step rule for each Hessian: 2/(L+mu)
        eta = 2 / (oracle.info.L + oracle.info.mu)
        history = GD(oracle, step=eta, max_iter=10000, tol=1e-7).run(x0)
        x = history.x[-1]
        eig = np.linalg.eigvalsh(oracle.A)
        kappa = float(eig[-1] / eig[0])
        original_gap = float(base.f(x) - base.info.f_star)
        rows.append(dict(alpha=alpha, condition_number=kappa,
                         iterations=len(history.f)-1,
                         converged=history.grad_norm[-1] <= 1e-7,
                         final_gradient_norm=history.grad_norm[-1],
                         original_objective=base.f(x),
                         original_objective_gap=original_gap,
                         regularized_objective=oracle.f(x),
                         weight_norm=np.linalg.norm(x)))
        ax.semilogy(np.maximum(history.grad_norm, 1e-16), label=f'alpha={alpha:g}')
    ax.set(xlabel='GD iterations', ylabel='Gradient norm (regularized objective)',
           title='GD convergence with L2 regularization')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / 'gd_convergence.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r['alpha'] for r in rows], [r['condition_number'] for r in rows], 'o-')
    ax.set(xlabel='Regularization alpha', ylabel='Hessian condition number',
           title='Effect of L2 regularization on conditioning')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / 'conditioning.png', dpi=160)
    plt.close(fig)
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"alpha={row['alpha']:g} kappa={row['condition_number']:.2f} "
              f"iterations={row['iterations']} gap={row['original_objective_gap']:.6g} "
              f"converged={row['converged']}")
    print('Saved results to', out)


if __name__ == '__main__':
    main()
