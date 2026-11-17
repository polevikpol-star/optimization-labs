"""ЛР3 / регуляризация: regularization trade-off on a fully constrained Markowitz problem.
Run from lab2/: python -m scripts.lab3_regularization_tradeoff
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oracles.constrained_markowitz import ConstrainedMarkowitzOracle

OUT = Path(__file__).resolve().parents[1] / 'results' / 'lab3_regularization'


def capped_simplex_projection(v, cap, tol=1e-13):
    """Euclidean projection onto {0 <= w_i <= cap, sum(w)=1}, via bisection."""
    v = np.asarray(v, dtype=float)
    if len(v) * cap < 1:
        raise ValueError('Empty capped simplex')
    lo, hi = float(np.min(v) - cap), float(np.max(v))
    for _ in range(100):
        mid = (lo + hi) / 2
        z = np.clip(v - mid, 0, cap)
        if z.sum() > 1:
            lo = mid
        else:
            hi = mid
        if hi-lo < tol:
            break
    return np.clip(v - (lo+hi)/2, 0, cap)


def feasible_projection(v, oracle):
    """Exact numerical Euclidean projection, enforcing budget, cap and return.

    No risk_max in this experiment. Risk is still measured in output.
    """
    cons = [{'type': 'eq', 'fun': lambda w: float(np.sum(w)-1),
             'jac': lambda w: np.ones_like(w)}]
    if oracle.r_min is not None:
        cons.append({'type': 'ineq', 'fun': lambda w: oracle.ret(w)-oracle.r_min,
                     'jac': lambda w: oracle.b.copy()})
    init = capped_simplex_projection(v, oracle.w_max)
    if oracle.is_feasible(init, tol=1e-10):
        return init
    res = minimize(lambda w: 0.5 * np.sum((w-v)**2), init,
                   jac=lambda w: w-v, method='SLSQP',
                   bounds=[(0, oracle.w_max)]*oracle.dim,
                   constraints=cons, options={'maxiter': 500, 'ftol': 1e-13})
    if not oracle.is_feasible(res.x, tol=1e-8):
        raise RuntimeError(f'Projection failed: {res.message}; violations={oracle.violations(res.x)}')
    return res.x


def projected_gd(oracle, alpha, w0, max_iter=3000, tol=1e-8):
    """Projected GD using step 1/(L+alpha), stopping by projected-gradient mapping."""
    L = float(np.linalg.eigvalsh(oracle.A)[-1] + alpha)
    eta = 1/L
    w = w0.copy()
    fvals = [oracle.f(w)]
    pg_norms = []
    for k in range(max_iter):
        grad = oracle.grad(w) + alpha*w
        next_w = feasible_projection(w - eta*grad, oracle)
        pg = np.linalg.norm(next_w-w) / eta
        pg_norms.append(pg)
        w = next_w
        fvals.append(oracle.f(w))
        if pg <= tol:
            break
    return w, k+1, pg_norms[-1], np.array(fvals)


def constrained_reference(oracle, alpha=0):
    """High accuracy constrained reference using SLSQP (not the unconstrained x_star)."""
    cons = [{'type': 'eq', 'fun': lambda w: w.sum()-1,
             'jac': lambda w: np.ones(oracle.dim)}]
    if oracle.r_min is not None:
        cons.append({'type': 'ineq', 'fun': lambda w: oracle.ret(w)-oracle.r_min,
                     'jac': lambda w: oracle.b.copy()})
    def obj(w): return oracle.f(w) + alpha/2*np.dot(w,w)
    def grad(w): return oracle.grad(w) + alpha*w
    x0 = oracle.feasible_start()
    if x0 is None:
        raise ValueError('No feasible start')
    res = minimize(obj, x0, jac=grad, method='SLSQP',
                   constraints=cons, bounds=[(0,oracle.w_max)]*oracle.dim,
                   options={'maxiter': 2000, 'ftol': 1e-13})
    if not oracle.is_feasible(res.x, 1e-8):
        raise RuntimeError(f'Reference infeasible: {res.message}')
    return res.x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seed, dim, n_periods, w_max = 0, 12, 300, 0.25
    prototype = ConstrainedMarkowitzOracle(dim=dim, n_periods=n_periods,
                                            seed=seed, w_max=w_max)
    equal_return = prototype.ret(np.full(dim, 1/dim))
    max_return = prototype.max_feasible_return()
    r_min = equal_return + 0.25*(max_return-equal_return)
    oracle = ConstrainedMarkowitzOracle(dim=dim, n_periods=n_periods,
                                        seed=seed, w_max=w_max, r_min=r_min)
    w0 = feasible_projection(np.full(dim, 1/dim), oracle)
    assert oracle.is_feasible(w0, 1e-8)
    w_star = constrained_reference(oracle, 0)
    f_star = oracle.f(w_star)
    eig = np.linalg.eigvalsh(oracle.A)
    alphas = [0, 1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3]
    epsilons = [0.03, 0.10, 0.25]
    rows = []
    histories = {}
    for alpha in alphas:
        w, iterations, pg, history = projected_gd(oracle, alpha, w0)
        w_alpha_ref = constrained_reference(oracle, alpha)
        orig_gap = oracle.f(w)-f_star
        distance = np.linalg.norm(w-w_star)
        viol = oracle.violations(w)
        row = dict(alpha=alpha, condition_number=(eig[-1]+alpha)/(eig[0]+alpha),
                   iterations=iterations, projected_gradient_mapping_norm=pg,
                   original_objective=oracle.f(w), original_optimality_gap=orig_gap,
                   modified_objective=oracle.f(w)+alpha/2*np.dot(w,w),
                   distance_to_original_constrained_optimum=distance,
                   distance_to_regularized_reference=np.linalg.norm(w-w_alpha_ref),
                   max_constraint_violation=viol['max'],
                   sum_violation=viol['sum'], nonneg_violation=viol['nonneg'],
                   cap_violation=viol['cap'], return_violation=viol['return'],
                   expected_return=oracle.ret(w), risk=oracle.risk(w),
                   converged=pg<=1e-8,
                   **{f'within_epsilon_{eps:g}': distance<=eps for eps in epsilons})
        rows.append(row)
        histories[alpha] = history
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'weights.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['asset','original_constrained_optimal_weight'])
        writer.writerows([(i, float(w_star[i])) for i in range(dim)])
    fig, ax=plt.subplots(figsize=(8,5))
    for a,h in histories.items():
        ax.plot(np.arange(len(h)), h, label=f'alpha={a:g}')
    ax.set(xlabel='Projected GD iterations', ylabel='Original f(w)',
           title='Original Markowitz objective throughout optimization')
    ax.legend();ax.grid(True);fig.tight_layout()
    fig.savefig(OUT/'convergence_original.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot([r['iterations'] for r in rows],
            [r['distance_to_original_constrained_optimum'] for r in rows], 'o-')
    for r in rows:
        ax.annotate(f"{r['alpha']:g}", (r['iterations'],r['distance_to_original_constrained_optimum']),xytext=(5,5),textcoords='offset points')
    for eps in epsilons:
        ax.axhline(eps,ls='--',alpha=.5,label=f'epsilon={eps:g}')
    ax.set(xlabel='Iterations to projected-gradient tolerance',
           ylabel='Distance to original constrained optimum (L2)',
           title='Speed vs. solution shift')
    ax.grid(True);ax.legend();fig.tight_layout();fig.savefig(OUT/'tradeoff.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot([r['alpha'] for r in rows],[r['original_optimality_gap'] for r in rows],'o-',label='Original f gap')
    ax.set(xlabel='Regularization alpha',ylabel='f(w_alpha)-f(w_star)',
           title='Degradation measured on original constrained objective')
    ax.grid(True);ax.legend();fig.tight_layout();fig.savefig(OUT/'original_gap.png',dpi=170);plt.close(fig)
    print(f'dim={dim}, seed={seed}, w_max={w_max}, r_min={r_min:.10g}')
    print('initial violations:',oracle.violations(w0))
    print(f'original constrained f_star={f_star:.12g}')
    for r in rows:
        print(f"alpha={r['alpha']:g} kappa={r['condition_number']:.3f} "
              f"iter={r['iterations']} pg={r['projected_gradient_mapping_norm']:.2e} "
              f"gap={r['original_optimality_gap']:.3e} "
              f"dist={r['distance_to_original_constrained_optimum']:.3e} "
              f"violation={r['max_constraint_violation']:.2e}")
    print('Outputs:',OUT)

if __name__=='__main__': main()
