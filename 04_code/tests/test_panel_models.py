"""Tests for the fixed-effect estimators, checked against known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from panel_models import (  # noqa: E402
    absorbed_ols,
    poisson_fe,
    poisson_score_bootstrap_wald,
    wald_test,
)


def _panel(n_units: int = 40, n_periods: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    cluster = unit // 2  # two units per cluster, as country x age band pairs
    return rng, unit, period, cluster


def test_absorbed_ols_recovers_a_known_coefficient():
    rng, unit, period, cluster = _panel()
    unit_effect = rng.normal(size=unit.max() + 1)[unit]
    period_effect = rng.normal(size=period.max() + 1)[period]
    treated = ((unit % 3 == 0) & (period >= 6)).astype(float)
    y = 2.0 + unit_effect + period_effect - 0.35 * treated + rng.normal(scale=0.05, size=len(unit))

    fit = absorbed_ols(y, treated[:, None], ["treated"], [unit, period], cluster)
    assert fit.names == ["treated"]
    assert fit.coef[0] == pytest.approx(-0.35, abs=0.02)
    assert fit.n_clusters == len(np.unique(cluster))


def test_absorbed_ols_drops_a_regressor_spanned_by_the_fixed_effects():
    _, unit, period, cluster = _panel()
    treated = ((unit % 3 == 0) & (period >= 6)).astype(float)
    # A pure unit-level regressor cannot survive the unit fixed effect.
    unit_constant = (unit % 3 == 0).astype(float)
    y = -0.4 * treated + np.random.default_rng(1).normal(size=len(unit))

    fit = absorbed_ols(
        y, np.column_stack([treated, unit_constant]), ["treated", "unit_constant"],
        [unit, period], cluster,
    )
    assert "unit_constant" not in fit.names
    assert "unit_constant" in fit.diagnostics["dropped_collinear_terms"]


def test_poisson_fe_recovers_a_known_rate_ratio():
    rng, unit, period, cluster = _panel(n_units=60, n_periods=14, seed=11)
    offset = np.log(rng.uniform(5e4, 5e5, size=unit.max() + 1))[unit]
    unit_effect = rng.normal(scale=0.3, size=unit.max() + 1)[unit]
    period_effect = np.linspace(0, -0.2, period.max() + 1)[period]
    treated = ((unit % 4 == 0) & (period >= 7)).astype(float)
    log_mu = offset - 9.0 + unit_effect + period_effect + np.log(0.8) * treated
    y = rng.poisson(np.exp(log_mu)).astype(float)

    fit = poisson_fe(
        y=y, X=np.column_stack([treated, np.eye(period.max() + 1)[period][:, 1:]]),
        names=["treated"] + [f"t{i}" for i in range(1, period.max() + 1)],
        offset=offset, absorb_unit=unit, cluster=cluster,
    )
    assert fit.converged
    assert np.exp(fit.coef[0]) == pytest.approx(0.8, abs=0.05)


def test_poisson_fe_accepts_non_integer_outcomes():
    """GBD supplies modelled case means, not counts; PML must still fit."""
    rng, unit, period, cluster = _panel(n_units=30, n_periods=10, seed=3)
    offset = np.zeros(len(unit))
    treated = ((unit % 3 == 0) & (period >= 5)).astype(float)
    y = np.exp(1.5 + 0.4 * rng.normal(size=len(unit)) - 0.25 * treated)
    assert not np.allclose(y, np.round(y))

    fit = poisson_fe(
        y=y, X=treated[:, None], names=["treated"], offset=offset,
        absorb_unit=unit, cluster=cluster,
    )
    assert fit.converged
    assert fit.coef[0] == pytest.approx(-0.25, abs=0.15)


def test_poisson_fe_drops_terms_collinear_with_the_unit_effect():
    _, unit, period, cluster = _panel(n_units=24, n_periods=8, seed=5)
    treated = ((unit % 2 == 0) & (period >= 4)).astype(float)
    unit_constant = (unit % 2 == 0).astype(float)
    y = np.full(len(unit), 10.0)

    fit = poisson_fe(
        y=y, X=np.column_stack([treated, unit_constant]),
        names=["treated", "unit_constant"], offset=np.zeros(len(unit)),
        absorb_unit=unit, cluster=cluster,
    )
    assert "unit_constant" in fit.diagnostics["dropped_collinear_terms"]
    assert "unit_constant" not in fit.names


def _poisson_fit_for_bootstrap(effect: float, seed: int = 11):
    rng, unit, period, cluster = _panel(n_units=60, n_periods=14, seed=seed)
    offset = np.log(rng.uniform(5e4, 5e5, size=unit.max() + 1))[unit]
    unit_effect = rng.normal(scale=0.3, size=unit.max() + 1)[unit]
    treated = ((unit % 4 == 0) & (period >= 7)).astype(float)
    log_mu = offset - 9.0 + unit_effect + np.log1p(effect) * treated
    y = rng.poisson(np.exp(log_mu)).astype(float)
    return poisson_fe(
        y=y, X=np.column_stack([treated, np.eye(period.max() + 1)[period][:, 1:]]),
        names=["treated"] + [f"t{i}" for i in range(1, period.max() + 1)],
        offset=offset, absorb_unit=unit, cluster=cluster,
    )


def test_poisson_score_bootstrap_does_not_reject_a_true_null():
    """``treated`` has no effect here, so the bootstrap must not reject it."""
    fit = _poisson_fit_for_bootstrap(effect=0.0)
    result = poisson_score_bootstrap_wald(fit, ["treated"], n_bootstrap=399)
    assert "score" in result["method"]
    assert result["restricted_fit_converged"] is True
    assert result["n_bootstrap"] == 399
    assert result["bootstrap_p_value"] > 0.05


def test_poisson_score_bootstrap_rejects_a_large_true_effect():
    fit = _poisson_fit_for_bootstrap(effect=-0.35)
    result = poisson_score_bootstrap_wald(fit, ["treated"], n_bootstrap=399)
    assert result["bootstrap_p_value"] < 0.05


def test_poisson_score_bootstrap_reports_rather_than_guesses_on_unknown_terms():
    fit = _poisson_fit_for_bootstrap(effect=0.0)
    result = poisson_score_bootstrap_wald(fit, ["not_a_term"], n_bootstrap=99)
    assert result["bootstrap_p_value"] is None
    assert result["terms"] == []


def test_poisson_score_bootstrap_refuses_a_fit_without_retained_state():
    fit = _poisson_fit_for_bootstrap(effect=0.0)
    fit.poisson_state = None
    with pytest.raises(ValueError, match="did not retain its Poisson state"):
        poisson_score_bootstrap_wald(fit, ["treated"], n_bootstrap=9)


def test_poisson_score_bootstrap_checks_the_restriction_shape():
    fit = _poisson_fit_for_bootstrap(effect=0.0)
    with pytest.raises(ValueError, match="columns but the design has"):
        poisson_score_bootstrap_wald(
            fit, ["treated"], restriction=np.ones((1, 3)), n_bootstrap=9
        )


def test_wald_test_is_never_negative_on_a_singular_covariance():
    """A rank-deficient cluster-robust covariance must not invert into noise."""
    beta = np.array([0.4, 0.4, 0.4])
    # Rank one: three coefficients identified by a single cluster's variation.
    direction = np.array([1.0, 1.0, 1.0])[:, None]
    covariance = direction @ direction.T * 0.05
    result = wald_test(beta, covariance)
    assert result["statistic"] >= 0
    assert result["df"] == 1
    assert result["rank_deficient"] is True
    assert 0.0 <= result["p_value"] <= 1.0


def test_wald_test_matches_the_analytic_value_when_well_conditioned():
    beta = np.array([1.0, 2.0])
    covariance = np.diag([1.0, 4.0])
    result = wald_test(beta, covariance)
    assert result["statistic"] == pytest.approx(1.0 / 1.0 + 4.0 / 4.0)
    assert result["df"] == 2
    assert result["rank_deficient"] is False
