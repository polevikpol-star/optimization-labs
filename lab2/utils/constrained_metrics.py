import numpy as np


def objective_value(f, x: np.ndarray) -> float:
    """Value of the original objective function."""
    return float(f(x))


def grad_norm(grad, x: np.ndarray) -> float:
    """Euclidean norm of the gradient."""
    return float(np.linalg.norm(grad(x)))


def budget_violation(x: np.ndarray, budget: float = 1.0) -> float:
    """Violation of sum(x) = budget."""
    return float(abs(np.sum(x) - budget))


def nonnegativity_violation(x: np.ndarray) -> float:
    """Violation of x_i >= 0."""
    return float(max(0.0, -np.min(x)))


def max_weight_violation(
    x: np.ndarray,
    max_weight: float,
) -> float:
    """Violation of x_i <= max_weight."""
    return float(max(0.0, np.max(x) - max_weight))


def portfolio_return(
    x: np.ndarray,
    mu: np.ndarray,
) -> float:
    """Expected portfolio return mu^T x."""
    return float(mu @ x)


def min_return_violation(
    x: np.ndarray,
    mu: np.ndarray,
    min_return: float,
) -> float:
    """Violation of mu^T x >= min_return."""
    return float(
        max(0.0, min_return - portfolio_return(x, mu))
    )


def portfolio_risk(
    x: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Portfolio variance x^T Sigma x."""
    return float(x @ sigma @ x)


def max_risk_violation(
    x: np.ndarray,
    sigma: np.ndarray,
    max_risk: float,
) -> float:
    """Violation of x^T Sigma x <= max_risk."""
    return float(
        max(0.0, portfolio_risk(x, sigma) - max_risk)
    )


def constraint_violations(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    max_weight: float | None = None,
    min_return: float | None = None,
    max_risk: float | None = None,
    budget: float = 1.0,
) -> dict:
    """Compute violations of all enabled constraints."""

    out = {
        "budget": budget_violation(x, budget),
        "nonnegativity": nonnegativity_violation(x),
    }

    if max_weight is not None:
        out["max_weight"] = max_weight_violation(
            x, max_weight
        )

    if min_return is not None:
        out["min_return"] = min_return_violation(
            x, mu, min_return
        )

    if max_risk is not None:
        out["max_risk"] = max_risk_violation(
            x, sigma, max_risk
        )

    return out


def max_constraint_violation(
    violations: dict,
) -> float:
    """Maximum violation among all constraints."""
    if not violations:
        return 0.0

    return float(max(violations.values()))


def is_feasible(
    violations: dict,
    tol: float = 1e-8,
) -> bool:
    """Check whether all constraints hold within tolerance."""
    return max_constraint_violation(violations) <= tol


def distance_to_reference(
    x: np.ndarray,
    x_ref: np.ndarray,
) -> float:
    """Distance to a reference constrained solution."""
    return float(np.linalg.norm(x - x_ref))


def projected_gradient_norm(
    x: np.ndarray,
    grad,
    projection,
    step: float,
) -> float:
    """
    Norm of projected-gradient mapping:
    ||x - P_C(x - step * grad(x))|| / step.
    """
    g = grad(x)
    x_projected = projection(x - step * g)

    mapping = (x - x_projected) / step

    return float(np.linalg.norm(mapping))


def summarize_constrained_solution(
    x: np.ndarray,
    f,
    grad,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_iter: int | None = None,
    max_weight: float | None = None,
    min_return: float | None = None,
    max_risk: float | None = None,
    budget: float = 1.0,
    x_ref: np.ndarray | None = None,
    tol: float = 1e-8,
) -> dict:
    """Summary of solution quality for constrained Markowitz."""

    violations = constraint_violations(
        x=x,
        mu=mu,
        sigma=sigma,
        max_weight=max_weight,
        min_return=min_return,
        max_risk=max_risk,
        budget=budget,
    )

    out = {
        "n_iter": n_iter,
        "f_original": objective_value(f, x),
        "grad_norm": grad_norm(grad, x),
        "portfolio_return": portfolio_return(x, mu),
        "portfolio_risk": portfolio_risk(x, sigma),
        "max_constraint_violation":
            max_constraint_violation(violations),
        "feasible": is_feasible(violations, tol),
        "constraint_violations": violations,
    }

    if x_ref is not None:
        out["distance_to_reference"] = (
            distance_to_reference(x, x_ref)
        )

    return out