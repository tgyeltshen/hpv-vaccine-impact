import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import power as P  # noqa: E402


def test_variance_is_the_sum_of_reciprocal_event_counts():
    assert P.log_rate_variance(100, 100, 100, 100) == pytest.approx(0.04)
    # A cell with no events carries no information.
    assert P.log_rate_variance(100, 0, 100, 100) == float("inf")
    assert P.log_rate_variance(-1, 100, 100, 100) == float("inf")


def test_variance_falls_as_cases_accumulate():
    small = P.log_rate_variance(100, 100, 1e6, 1e6)
    large = P.log_rate_variance(10_000, 10_000, 1e6, 1e6)
    assert large < small
    # Quadrupling the cases halves the standard error.
    assert np.sqrt(P.log_rate_variance(400, 1e9, 1e9, 1e9)) == pytest.approx(
        np.sqrt(P.log_rate_variance(100, 1e9, 1e9, 1e9)) / 2, rel=1e-3
    )


def test_minimum_detectable_effect_matches_the_textbook_multiplier():
    mde = P.minimum_detectable_effect(0.05)
    assert mde["log_scale"] == pytest.approx((1.959963984540054 + 0.8416212335729143) * 0.05)
    assert mde["irr"] == pytest.approx(np.exp(-mde["log_scale"]))
    assert mde["reduction_pct"] == pytest.approx(100 * (1 - mde["irr"]))

    # Bigger standard error, bigger effect needed.
    assert P.minimum_detectable_effect(0.10)["reduction_pct"] > mde["reduction_pct"]
    assert P.minimum_detectable_effect(float("inf"))["reduction_pct"] == 100.0


def test_design_effect_and_effective_cases_are_consistent():
    """A design effect of 2 wastes three quarters of the information."""
    assert P.design_effect(0.06, 0.03) == pytest.approx(2.0)
    assert P.effective_cases(10_000, 2.0) == pytest.approx(2_500)
    # No clustering penalty means no information lost.
    assert P.effective_cases(10_000, 1.0) == pytest.approx(10_000)
    assert np.isnan(P.design_effect(0.06, 0.0))


def test_expected_reduction_is_the_product_and_rejects_non_proportions():
    assert P.expected_itt_reduction(0.7, 0.9) == pytest.approx(0.63)
    assert P.expected_itt_reduction(0.7, 0.9, 0.5) == pytest.approx(0.315)
    assert P.expected_itt_reduction(0.0, 0.9) == 0.0
    for bad in ((1.2, 0.9), (0.7, -0.1)):
        with pytest.raises(ValueError):
            P.expected_itt_reduction(*bad)


def test_a_well_powered_design_has_an_mde_below_the_expected_effect():
    """The comparison the paper turns on, stated as an executable check."""
    expected = 100 * P.expected_itt_reduction(0.7, 0.9)
    plentiful = P.minimum_detectable_effect(
        np.sqrt(P.log_rate_variance(5_000, 5_000, 1e6, 1e6))
    )
    assert plentiful["reduction_pct"] < expected

    # Eight cases a side - the order of magnitude of a non-Brazil treated cell.
    scarce = P.minimum_detectable_effect(
        np.sqrt(P.log_rate_variance(8, 8, 1e6, 1e6))
    )
    assert scarce["reduction_pct"] > expected
