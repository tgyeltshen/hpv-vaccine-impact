"""Fixed-effect panel estimators used by the analysis stage.

Two estimators are implemented here rather than taken from a library:

* :func:`absorbed_ols` — linear regression with two absorbed fixed-effect sets
  and cluster-robust standard errors. This is the engine for the Sun-Abraham
  interaction-weighted estimator and for the labelled two-way fixed-effect
  comparison.
* :func:`poisson_fe` — Poisson pseudo-maximum-likelihood with one absorbed
  high-dimensional fixed effect (the unit), explicit remaining dummies, an
  exposure offset, and cluster-robust standard errors. This is the count model
  the protocol specifies as primary.

Both return cluster-robust variance matrices, because the protocol clusters
sampling uncertainty at country while the unit of treatment is country x age
band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    names: list[str]
    coef: np.ndarray
    vcov: np.ndarray
    n_obs: int
    n_clusters: int
    converged: bool = True
    iterations: int = 0
    diagnostics: dict = field(default_factory=dict)
    demeaned: tuple | None = field(default=None, repr=False)
    poisson_state: dict | None = field(default=None, repr=False)

    @property
    def std_error(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.vcov), 0.0, None))

    def as_table(self, alpha: float = 0.05) -> list[dict[str, object]]:
        from scipy import stats

        critical = stats.norm.ppf(1 - alpha / 2)
        se = self.std_error
        rows = []
        for index, name in enumerate(self.names):
            estimate, error = float(self.coef[index]), float(se[index])
            rows.append(
                {
                    "term": name,
                    "estimate": estimate,
                    "std_error": error,
                    "conf_low": estimate - critical * error,
                    "conf_high": estimate + critical * error,
                }
            )
        return rows

    def wald(self, terms: list[str]) -> dict[str, object]:
        """Joint test that the named coefficients are all zero."""
        from scipy import stats

        index = [self.names.index(term) for term in terms if term in self.names]
        if not index:
            return {"terms": [], "statistic": None, "df": 0, "p_value": None}
        selected = np.asarray(index)
        result = wald_test(self.coef[selected], self.vcov[np.ix_(selected, selected)])
        result["terms"] = [self.names[i] for i in selected]
        return result


def wald_test(beta: np.ndarray, covariance: np.ndarray) -> dict[str, object]:
    """Joint zero test using a spectrally truncated generalised inverse.

    A cluster-robust covariance built from few effective clusters is only
    positive *semi*-definite, and here it is routinely near-singular: several
    treatment cohorts contain a single country, so their coefficients share one
    cluster's variation. A plain pseudo-inverse then inverts numerical noise and
    can return a negative chi-square. Eigenvalues at or below the tolerance are
    dropped instead, and the surviving rank is reported as the degrees of
    freedom so the test is read for what it is: a test on the well-identified
    subspace, not on every term named.
    """
    from scipy import stats

    beta = np.asarray(beta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    symmetric = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    largest = float(values.max()) if values.size else 0.0
    if largest <= 0:
        return {"statistic": None, "df": 0, "p_value": None, "rank_deficient": True}
    tolerance = largest * len(values) * np.finfo(float).eps * 1e4
    keep = values > tolerance
    rank = int(keep.sum())
    if rank == 0:
        return {"statistic": None, "df": 0, "p_value": None, "rank_deficient": True}
    projected = vectors[:, keep].T @ beta
    statistic = float(np.sum(projected ** 2 / values[keep]))
    return {
        "statistic": statistic,
        "df": rank,
        "p_value": float(stats.chi2.sf(statistic, rank)),
        "rank_deficient": bool(rank < len(beta)),
        "terms_tested": int(len(beta)),
    }


def _group_codes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(labels, return_inverse=True)
    return codes.astype(np.int64), int(codes.max()) + 1


def _demean(
    matrix: np.ndarray,
    factors: list[tuple[np.ndarray, int]],
    weights: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> np.ndarray:
    """Remove several fixed-effect sets by alternating weighted projections."""
    out = np.array(matrix, dtype=float, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    w = np.ones(out.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    denominators = [np.bincount(codes, weights=w, minlength=size) for codes, size in factors]
    for iteration in range(max_iterations):
        largest = 0.0
        for (codes, size), denominator in zip(factors, denominators):
            for column in range(out.shape[1]):
                numerator = np.bincount(codes, weights=w * out[:, column], minlength=size)
                means = np.divide(
                    numerator, denominator,
                    out=np.zeros_like(numerator), where=denominator > 0,
                )
                shift = means[codes]
                out[:, column] -= shift
                largest = max(largest, float(np.abs(shift).max(initial=0.0)))
        if largest < tolerance:
            return out
    raise RuntimeError(
        f"fixed-effect absorption did not converge in {max_iterations} sweeps "
        f"(last shift {largest:.2e})"
    )


def _independent_columns(matrix: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Indices of a maximal linearly independent subset of columns.

    Used to drop regressors that the absorbed fixed effects already span. For
    an age-by-year effect set this is not a rare edge case: every unit holds one
    age band for its whole life, so the year dummies of that band sum to one
    within the unit and one column per age band is redundant with the unit
    effect.
    """
    from scipy.linalg import qr

    if matrix.shape[1] == 0:
        return np.array([], dtype=int)
    _, r, pivot = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.array([], dtype=int)
    rank = int(np.sum(diagonal > tolerance * diagonal[0] * max(matrix.shape)))
    return np.sort(pivot[:rank])


def _cluster_sandwich(
    bread: np.ndarray, scores: np.ndarray, cluster_codes: np.ndarray, n_clusters: int,
    n_obs: int, n_params: int,
) -> np.ndarray:
    totals = np.zeros((n_clusters, scores.shape[1]))
    np.add.at(totals, cluster_codes, scores)
    meat = totals.T @ totals
    # Standard small-sample correction, as used by common cluster-robust
    # implementations. With few clusters it is a mild adjustment, not a fix.
    correction = (n_clusters / max(n_clusters - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_params, 1)
    )
    inverse = np.linalg.pinv(bread)
    return correction * inverse @ meat @ inverse


def absorbed_ols(
    y: np.ndarray,
    X: np.ndarray,
    names: list[str],
    absorb: list[np.ndarray],
    cluster: np.ndarray,
) -> FitResult:
    """OLS with absorbed fixed effects and cluster-robust standard errors."""
    factors = [_group_codes(a) for a in absorb]
    cluster_codes, n_clusters = _group_codes(cluster)

    y_tilde = _demean(y, factors).ravel()
    x_tilde = _demean(X, factors)

    # Regressors made collinear by the absorbed effects are dropped rather than
    # silently inverted through a pseudo-inverse.
    keep = np.abs(x_tilde).max(axis=0) > 1e-9
    x_used, used_names = x_tilde[:, keep], [n for n, k in zip(names, keep) if k]
    q, r = np.linalg.qr(x_used)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-9 * max(1.0, abs(r[0, 0]))))
    if rank < x_used.shape[1]:
        independent = np.abs(np.diag(r)) > 1e-9 * max(1.0, abs(r[0, 0]))
        x_used = x_used[:, independent]
        used_names = [n for n, k in zip(used_names, independent) if k]

    bread = x_used.T @ x_used
    coef = np.linalg.solve(bread, x_used.T @ y_tilde)
    residual = y_tilde - x_used @ coef
    n_absorbed = sum(size for _, size in factors)
    vcov = _cluster_sandwich(
        bread, x_used * residual[:, None], cluster_codes, n_clusters,
        len(y_tilde), x_used.shape[1] + n_absorbed,
    )
    dropped = [n for n in names if n not in used_names]
    result = FitResult(
        names=used_names, coef=coef, vcov=vcov, n_obs=len(y_tilde),
        n_clusters=n_clusters,
        diagnostics={"dropped_collinear_terms": dropped, "absorbed_levels": n_absorbed},
    )
    # Kept so a wild cluster bootstrap can resample without redoing the
    # absorption: once y and X are demeaned, the model is an ordinary linear
    # one and a bootstrap replication is a matrix-vector product.
    result.demeaned = (y_tilde, x_used, cluster_codes, n_clusters)
    return result


def wild_cluster_bootstrap_wald(
    fit: FitResult,
    test_terms: list[str],
    weights: np.ndarray | None = None,
    n_bootstrap: int = 1999,
    seed: int = 20260812,
) -> dict[str, object]:
    """Restricted wild cluster bootstrap p-value for a joint zero test.

    The asymptotic cluster-robust Wald test over-rejects badly when few
    clusters carry the tested variation, which is exactly this panel's
    situation: 23 treated countries identify every lead and lag. Under the null
    the tested coefficients are set to zero, residuals are resampled with
    cluster-level Rademacher weights, and the test statistic is recomputed, so
    the p-value is calibrated against the actual cluster structure rather than
    against a chi-square limit that has not arrived.

    ``weights`` optionally maps coefficients onto the aggregated quantities
    being tested (the interaction-weighted event-time estimates).
    """
    if fit.demeaned is None:
        raise ValueError("this fit did not retain its demeaned design")
    y, X, cluster_codes, n_clusters = fit.demeaned
    index = [fit.names.index(term) for term in test_terms if term in fit.names]
    if not index:
        return {"terms": [], "bootstrap_p_value": None, "n_bootstrap": 0}
    tested = np.asarray(index)
    free = np.array([i for i in range(X.shape[1]) if i not in set(index)], dtype=int)

    def statistic(response: np.ndarray) -> float:
        bread = X.T @ X
        coef = np.linalg.solve(bread, X.T @ response)
        residual = response - X @ coef
        vcov = _cluster_sandwich(
            bread, X * residual[:, None], cluster_codes, n_clusters,
            len(response), X.shape[1],
        )
        if weights is None:
            beta, covariance = coef[tested], vcov[np.ix_(tested, tested)]
        else:
            beta, covariance = weights @ coef, weights @ vcov @ weights.T
        outcome = wald_test(beta, covariance)
        return outcome["statistic"] if outcome["statistic"] is not None else np.nan

    observed = statistic(y)

    # Restricted fit: the tested coefficients are held at zero.
    if free.size:
        restricted_coef = np.linalg.solve(X[:, free].T @ X[:, free], X[:, free].T @ y)
        fitted = X[:, free] @ restricted_coef
    else:
        fitted = np.zeros_like(y)
    restricted_residual = y - fitted

    rng = np.random.default_rng(seed)
    exceed, completed = 0, 0
    for _ in range(n_bootstrap):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        draw = fitted + signs[cluster_codes] * restricted_residual
        value = statistic(draw)
        if not np.isfinite(value):
            continue
        completed += 1
        if value >= observed:
            exceed += 1
    return {
        "terms": [fit.names[i] for i in tested],
        "observed_statistic": float(observed),
        "bootstrap_p_value": (exceed + 1) / (completed + 1) if completed else None,
        "n_bootstrap": completed,
        "method": "restricted wild cluster bootstrap, Rademacher weights",
    }


def poisson_fe(
    y: np.ndarray,
    X: np.ndarray,
    names: list[str],
    offset: np.ndarray,
    absorb_unit: np.ndarray,
    cluster: np.ndarray,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> FitResult:
    """Poisson PML with the unit effect concentrated out of the likelihood.

    The unit effect has a closed-form solution given the other parameters, so it
    is profiled rather than estimated as 792 dummies. By the envelope theorem the
    score of the concentrated likelihood is ``X'(y - mu)``, and its Hessian is
    ``X~'W X~`` where ``X~`` is ``X`` demeaned within unit using the Poisson
    weights ``mu``. The outcome may be non-integer: GBD supplies modelled case
    means, and Poisson PML is a quasi-likelihood that does not require counts.
    """
    unit_codes, n_units = _group_codes(absorb_unit)
    cluster_codes, n_clusters = _group_codes(cluster)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    offset = np.asarray(offset, dtype=float)

    unit_totals = np.bincount(unit_codes, weights=y, minlength=n_units)
    if np.any(unit_totals <= 0):
        raise ValueError("a unit has no outcome mass; its effect is not identified")

    keep = _independent_columns(_demean(X, [(unit_codes, n_units)]))
    dropped = [name for index, name in enumerate(names) if index not in set(keep.tolist())]
    X = X[:, keep]
    names = [names[index] for index in keep]

    beta = np.zeros(X.shape[1])
    converged, iteration = False, 0
    for iteration in range(1, max_iterations + 1):
        linear = offset + X @ beta
        # Profiled unit effect: exp(alpha_i) = sum_t y_it / sum_t exp(linear_it)
        scale = np.bincount(unit_codes, weights=np.exp(linear), minlength=n_units)
        alpha = np.log(unit_totals) - np.log(scale)
        mu = np.exp(linear + alpha[unit_codes])

        x_tilde = _demean(X, [(unit_codes, n_units)], weights=mu)
        hessian = x_tilde.T @ (mu[:, None] * x_tilde)
        score = x_tilde.T @ (y - mu)
        step = np.linalg.solve(hessian, score)
        # A halved step keeps the update inside the region where exp() is finite
        # when a starting value is far from the optimum.
        scale_factor = 1.0
        while scale_factor > 1e-4:
            candidate = beta + scale_factor * step
            if np.max(np.abs(offset + X @ candidate)) < 500:
                break
            scale_factor /= 2
        beta = beta + scale_factor * step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break

    linear = offset + X @ beta
    scale = np.bincount(unit_codes, weights=np.exp(linear), minlength=n_units)
    alpha = np.log(unit_totals) - np.log(scale)
    mu = np.exp(linear + alpha[unit_codes])
    x_tilde = _demean(X, [(unit_codes, n_units)], weights=mu)
    hessian = x_tilde.T @ (mu[:, None] * x_tilde)
    vcov = _cluster_sandwich(
        hessian, x_tilde * (y - mu)[:, None], cluster_codes, n_clusters,
        len(y), X.shape[1] + n_units,
    )
    deviance_weights = np.where(y > 0, y * np.log(np.where(y > 0, y / mu, 1.0)), 0.0)
    result = FitResult(
        names=list(names), coef=beta, vcov=vcov, n_obs=len(y), n_clusters=n_clusters,
        converged=converged, iterations=iteration,
        diagnostics={
            "absorbed_units": n_units,
            "dropped_collinear_terms": dropped,
            "deviance": float(2 * np.sum(deviance_weights - (y - mu))),
            "pearson_dispersion": float(np.sum((y - mu) ** 2 / mu) / (len(y) - X.shape[1] - n_units)),
        },
    )
    # Kept so a score bootstrap can impose a null and recompute cluster score
    # contributions without the caller having to rebuild the design. ``X`` and
    # ``names`` here are already past the collinearity drop, so they line up with
    # ``result.names`` and with any restriction matrix expressed in that space.
    result.poisson_state = {
        "y": y, "X": X, "offset": offset,
        "absorb_unit": np.asarray(absorb_unit), "cluster": np.asarray(cluster),
        "unit_codes": unit_codes, "n_units": n_units,
        "cluster_codes": cluster_codes, "n_clusters": n_clusters,
    }
    return result


def poisson_score_bootstrap_wald(
    fit: FitResult,
    test_terms: list[str],
    restriction: np.ndarray | None = None,
    n_bootstrap: int = 1999,
    seed: int = 20260812,
) -> dict[str, object]:
    """Restricted score (LM) wild cluster bootstrap for a Poisson PML fit.

    The linear engine's bootstrap resamples the outcome and refits, which is not
    available here: a Poisson likelihood needs a non-negative outcome, and
    ``fitted + sign * residual`` does not respect that, quite apart from the cost
    of refitting two thousand times. A robust score bootstrap avoids both
    problems. The null is imposed once, and each replication only reweights the
    per-cluster score contributions, which is a handful of matrix products.

    Writing ``H`` for the Hessian of the concentrated likelihood at the
    restricted estimate, ``S_g`` for cluster ``g``'s score sum, and ``R`` for the
    tested linear combinations, the statistic is a quadratic form in
    ``d = sum_g R H^-1 S_g`` with cluster-robust variance ``sum_g d_g d_g'``.
    Rademacher cluster weights leave that variance unchanged, so it is formed
    once. ``restriction`` supplies ``R`` in the space of ``fit.names``, which is
    how the interaction-weighted event-time aggregates are tested; when it is
    omitted the tested coefficients are tested individually.
    """
    if fit.poisson_state is None:
        raise ValueError("this fit did not retain its Poisson state")
    state = fit.poisson_state
    y, X, offset = state["y"], state["X"], state["offset"]
    unit_codes, n_units = state["unit_codes"], state["n_units"]
    cluster_codes, n_clusters = state["cluster_codes"], state["n_clusters"]

    tested = [fit.names.index(term) for term in test_terms if term in fit.names]
    if not tested:
        return {"terms": [], "bootstrap_p_value": None, "n_bootstrap": 0}
    free = [index for index in range(X.shape[1]) if index not in set(tested)]

    # Impose the null: the tested coefficients are held at zero and the rest are
    # re-estimated. Any column the restricted fit drops as collinear stays zero,
    # which is the right value for a direction the restricted design cannot see.
    restricted = poisson_fe(
        y=y, X=X[:, free], names=[fit.names[index] for index in free],
        offset=offset, absorb_unit=state["absorb_unit"], cluster=state["cluster"],
    )
    beta = np.zeros(X.shape[1])
    position = {name: index for index, name in enumerate(fit.names)}
    for name, value in zip(restricted.names, restricted.coef):
        beta[position[name]] = value

    linear = offset + X @ beta
    unit_totals = np.bincount(unit_codes, weights=y, minlength=n_units)
    scale = np.bincount(unit_codes, weights=np.exp(linear), minlength=n_units)
    alpha = np.log(unit_totals) - np.log(scale)
    mu = np.exp(linear + alpha[unit_codes])

    x_tilde = _demean(X, [(unit_codes, n_units)], weights=mu)
    hessian = x_tilde.T @ (mu[:, None] * x_tilde)
    contributions = x_tilde * (y - mu)[:, None]
    cluster_scores = np.zeros((n_clusters, X.shape[1]))
    np.add.at(cluster_scores, cluster_codes, contributions)

    if restriction is None:
        selector = np.zeros((len(tested), X.shape[1]))
        selector[np.arange(len(tested)), tested] = 1.0
    else:
        selector = np.asarray(restriction, dtype=float)
        if selector.shape[1] != X.shape[1]:
            raise ValueError(
                "restriction has "
                f"{selector.shape[1]} columns but the design has {X.shape[1]}"
            )
    if not selector.size or not selector.any():
        return {"terms": [], "bootstrap_p_value": None, "n_bootstrap": 0}

    # d_g = R H^-1 S_g, stacked one row per cluster.
    per_cluster = cluster_scores @ np.linalg.pinv(hessian).T @ selector.T
    variance = per_cluster.T @ per_cluster
    observed_test = wald_test(per_cluster.sum(axis=0), variance)
    observed = observed_test["statistic"]
    if observed is None or not np.isfinite(observed):
        return {"terms": [fit.names[i] for i in tested],
                "bootstrap_p_value": None, "n_bootstrap": 0,
                "reason": "the observed score statistic was not computable"}

    rng = np.random.default_rng(seed)
    exceed, completed = 0, 0
    for _ in range(n_bootstrap):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        outcome = wald_test((signs[:, None] * per_cluster).sum(axis=0), variance)
        value = outcome["statistic"]
        if value is None or not np.isfinite(value):
            continue
        completed += 1
        if value >= observed:
            exceed += 1
    return {
        "terms": [fit.names[i] for i in tested],
        "observed_statistic": float(observed),
        "bootstrap_p_value": (exceed + 1) / (completed + 1) if completed else None,
        "n_bootstrap": completed,
        "restricted_fit_converged": bool(restricted.converged),
        "method": "restricted score (LM) wild cluster bootstrap, Rademacher weights",
    }
