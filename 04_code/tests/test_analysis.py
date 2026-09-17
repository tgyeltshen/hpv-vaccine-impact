"""Tests for the analysis stage on synthetic panels with known answers.

These use a purpose-built staggered panel rather than the project data, so the
expected values are derivable rather than whatever the pipeline last produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analysis as A  # noqa: E402


def synthetic_panel(
    effect: float = -0.30,
    pre_trend: float = 0.0,
    seed: int = 42,
    n_control_countries: int = 40,
    n_treated_countries: int = 24,
    noise: float = 0.05,
) -> pd.DataFrame:
    """Staggered panel with a known post-treatment effect on the log rate.

    The treated group is deliberately larger than the project's own 23
    countries. Cluster-robust inference is driven by the number of *treated*
    clusters, so a four-country toy panel would produce standard errors far too
    small to test any hypothesis against, regardless of the estimator.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1995, 2024)
    adoption_years = [2008, 2010, 2012, 2013, 2015, 2016]
    cohorts = {
        f"T{index:02d}": adoption_years[index % len(adoption_years)]
        for index in range(n_treated_countries)
    }
    rows = []
    for country, cohort in cohorts.items():
        level = rng.normal(1.5, 0.3)
        for year in years:
            event_time = year - cohort
            log_rate = (
                level
                - 0.01 * (year - years[0])
                + pre_trend * min(event_time, 0)
                + (effect if event_time >= 0 else 0.0)
                + rng.normal(scale=noise)
            )
            rows.append({
                "iso3": country, "age_group": "20-24", "year": year,
                "treatment_cohort_year": cohort, "ever_treated": 1,
                "treated": int(event_time >= 0), "event_time": float(event_time),
                "log_rate_true": log_rate,
                "analysis_status": "treated_documented" if event_time >= 0
                else "untreated_pre_eligibility_cohort",
                "historical_schedule_complete": 1,
            })
    for index in range(n_control_countries):
        country = f"C{index:02d}"
        level = rng.normal(1.5, 0.3)
        for year in years:
            rows.append({
                "iso3": country, "age_group": "20-24", "year": year,
                "treatment_cohort_year": 0, "ever_treated": 0, "treated": 0,
                "event_time": np.nan,
                "log_rate_true": level - 0.01 * (year - years[0]) + rng.normal(scale=noise),
                "analysis_status": "untreated_no_national_programme",
                "historical_schedule_complete": 1,
            })
    frame = pd.DataFrame(rows)
    frame["unit_id"] = frame["iso3"] + "|" + frame["age_group"]
    frame["cohort"] = frame["treatment_cohort_year"].replace(0, np.nan)
    frame["age_year"] = frame["age_group"] + ":" + frame["year"].astype(str)
    frame["log_rate"] = frame["log_rate_true"]
    frame["rate_per_100k"] = np.exp(frame["log_rate"])
    frame["gbd_population"] = 2.0e5
    frame["log_gbd_population"] = np.log(frame["gbd_population"])
    frame["cases"] = frame["rate_per_100k"] / 1e5 * frame["gbd_population"]
    return frame


def test_callaway_santanna_recovers_a_known_effect():
    """Recovery is asserted on a low-noise panel, deliberately.

    At the generator's default ``noise=0.05`` the realised sampling error of a
    24-country treated group is itself around 0.03 on the log scale, so a
    single-draw assertion at ``abs=0.03`` tests the luck of the seed rather than
    the estimator: seed 42 lands at -0.333 while the mean over seeds is -0.300.
    Shrinking the noise shrinks the sampling error, which is what makes a tight
    tolerance meaningful here. Unbiasedness across draws is asserted separately
    in ``test_estimators_are_unbiased_across_draws``.
    """
    frame = synthetic_panel(effect=-0.30, noise=0.005)
    result = A.callaway_santanna(frame, anticipation=0)
    overall = result["overall_att"][0]
    assert overall["estimate"] == pytest.approx(-0.30, abs=0.01)
    assert result["base_period"] == "universal"


def test_estimators_are_unbiased_across_draws():
    """Averaging over draws separates bias from one seed's sampling error."""
    cs, sa = [], []
    for seed in range(8):
        frame = synthetic_panel(effect=-0.30, pre_trend=0.0, seed=seed, noise=0.02)
        cs.append(A.callaway_santanna(frame, anticipation=0)["overall_att"][0]["estimate"])
        sa.append(A.sun_abraham(frame, 0, window=(-8, 6))["overall_post_att"])
    for label, values in (("Callaway-Sant'Anna", cs), ("Sun-Abraham", sa)):
        mean = float(np.mean(values))
        mc_se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        assert abs(mean - (-0.30)) < max(3 * mc_se, 0.01), (
            f"{label}: mean {mean:+.4f} is {abs(mean + 0.30) / max(mc_se, 1e-12):.1f} "
            f"Monte Carlo standard errors from the truth"
        )


def test_sun_abraham_recovers_a_known_effect_under_both_engines():
    frame = synthetic_panel(effect=-0.30)
    for engine in ("ols", "poisson"):
        result = A.sun_abraham(frame, anticipation=0, window=(-8, 6), engine=engine)
        assert result["overall_post_att"] == pytest.approx(-0.30, abs=0.05), engine
        assert result["overall_post_std_error"] > 0


def test_flat_pre_trend_is_not_rejected_and_a_real_one_is(monkeypatch):
    """The bootstrap is the decisive test, because the asymptotic one is oversized.

    With 24 treated clusters identifying every lead, the asymptotic
    cluster-robust Wald test rejects a panel built to satisfy parallel trends
    exactly in roughly 40% of draws at nominal 5%; the restricted wild cluster
    bootstrap rejects in roughly 5%. The decision rule in
    ``estimation_report.evaluate_decision_rules`` therefore reads the bootstrap,
    and so does this test.
    """
    monkeypatch.setattr(A, "BOOTSTRAP_REPLICATIONS", 399)

    flat = A.sun_abraham(synthetic_panel(pre_trend=0.0), 0, window=(-8, 6))
    assert flat["joint_pre_trend_bootstrap"]["bootstrap_p_value"] > 0.05

    # A real pre-trend must be caught by the correctly sized test too, or the
    # bootstrap would have bought size at the cost of all its power.
    sloped = A.sun_abraham(synthetic_panel(pre_trend=0.02), 0, window=(-8, 6))
    assert sloped["joint_pre_trend_bootstrap"]["bootstrap_p_value"] < 0.05
    assert sloped["joint_pre_trend_test"]["p_value"] < 0.05


def _panel_with_eligibility_ages(effect: float = -0.30) -> pd.DataFrame:
    """Synthetic panel carrying the fields the D4 restriction reads.

    Half the treated countries are given an adult age at eligibility so the
    restriction has something to remove.
    """
    frame = synthetic_panel(effect=effect, pre_trend=0.0)
    treated = frame["ever_treated"] == 1
    frame["primary_treated_eligible"] = (frame["treated"] == 1).astype(int)
    # Even-numbered treated countries reached young, odd ones only in adulthood.
    adult = treated & (frame["iso3"].str[1:].astype(int) % 2 == 1)
    # Float with NaN for the untreated, which is how pandas reads the blank
    # cells of this column out of the analytic dataset.
    frame["age_at_eligibility_max"] = np.where(treated, 12.0, np.nan)
    frame.loc[adult, "age_at_eligibility_max"] = 24.0
    return frame


def test_prophylactic_restriction_drops_only_the_adult_reached_units():
    frame = _panel_with_eligibility_ages()
    restricted = A.restrict_to_prophylactic_age(frame, 15)

    kept = set(restricted.loc[restricted["ever_treated"] == 1, "iso3"])
    assert kept, "the restriction must not empty the treated group"
    assert all(int(name[1:]) % 2 == 0 for name in kept), (
        "only cohorts reached young enough may remain treated"
    )
    # Units that lose treatment become controls rather than disappearing.
    assert len(restricted) == len(frame)
    dropped = restricted.loc[
        (frame["ever_treated"] == 1) & (restricted["ever_treated"] == 0)
    ]
    assert not dropped.empty
    assert dropped["event_time"].isna().all()
    assert (dropped["treated"] == 0).all()


def test_adult_negative_control_keeps_the_complement():
    frame = _panel_with_eligibility_ages()
    adult = A.restrict_to_adult_eligibility(frame, 20)
    kept = set(adult.loc[adult["ever_treated"] == 1, "iso3"])
    assert kept
    assert all(int(name[1:]) % 2 == 1 for name in kept)


def test_in_time_placebo_removes_every_real_post_period_observation():
    frame = _panel_with_eligibility_ages()
    pseudo = A.placebo_in_time(frame, 8)

    treated_rows = frame.loc[frame["ever_treated"] == 1, ["iso3", "age_group", "year"]]
    real_cohort = dict(
        frame.loc[frame["ever_treated"] == 1]
        .groupby("iso3")["treatment_cohort_year"].first()
    )
    for iso3, cohort in real_cohort.items():
        block = pseudo[pseudo["iso3"] == iso3]
        assert (block["year"] < cohort).all(), (
            "no genuinely post-eligibility year may survive the in-time placebo"
        )
        # The pseudo onset sits exactly lead_years before the real one.
        assert block["treatment_cohort_year"].max() == cohort - 8
    assert len(treated_rows) > 0
    # Controls are untouched, so the comparison group is unchanged.
    assert (pseudo["ever_treated"] == 0).sum() > 0


def test_placebo_on_a_flat_panel_finds_nothing(monkeypatch):
    """Guards against the placebo machinery inventing an effect by itself."""
    monkeypatch.setattr(A, "BOOTSTRAP_REPLICATIONS", 0)
    # No treatment effect and no pre-trend: the placebo must come back null.
    frame = _panel_with_eligibility_ages(effect=0.0)
    pseudo = A.placebo_in_time(frame, 8)
    result = A.sun_abraham(
        pseudo, 0, window=(-8, 6), engine="ols", bootstrap_replications=0
    )
    zero = next(r for r in result["event_study"] if r["event_time"] == 0)
    assert zero["conf_low"] < 0 < zero["conf_high"], (
        f"placebo found an effect on a flat panel: {zero}"
    )


def test_the_two_treated_cell_counts_reconcile():
    """The absorbing count must equal the documented count plus the difference.

    Guards the gap that made 104 and 106 look like an inconsistency: the sample
    block has to state both definitions and account for every cell between them,
    so no reader has to reverse-engineer which number is right.
    """
    frame = synthetic_panel(effect=-0.30)
    # One post-treatment cell is documented-ambiguous while still being treated
    # under the absorbing rule, which is the real panel's situation.
    mask = (frame["iso3"] == "T00") & (frame["event_time"] == 3)
    frame.loc[mask, "analysis_status"] = "ambiguous_partial_cohort_overlap"

    counts = A.treated_cell_counts(frame)
    assert (
        counts["treated_cell_count"]
        == counts["treated_cells_documented"] + counts["treated_cells_absorbing_only"]
    )
    assert counts["treated_cells_absorbing_only"] == int(mask.sum())
    assert len(counts["treated_cells_absorbing_only_detail"]) == int(mask.sum())
    assert {row["iso3"] for row in counts["treated_cells_absorbing_only_detail"]} == {"T00"}


def test_treated_cell_counts_agree_when_nothing_is_ambiguous():
    counts = A.treated_cell_counts(synthetic_panel(effect=-0.30))
    assert counts["treated_cells_absorbing_only"] == 0
    assert counts["treated_cells_absorbing_only_detail"] == []
    assert counts["treated_cell_count"] == counts["treated_cells_documented"]


def test_case_concentration_shares_reconcile():
    """The shares must add up and the leading country must be found, not assumed."""
    panel = synthetic_panel(effect=-0.30)
    concentration = A.treated_case_concentration(panel)

    shares = [row["share"] for row in concentration["treated_cases_by_country"]]
    assert sum(shares) == pytest.approx(1.0)
    assert shares == sorted(shares, reverse=True)
    assert concentration["treated_cases_top_country_share"] == pytest.approx(shares[0])

    top = concentration["treated_cases_top_country"]
    by_country = {row["iso3"]: row["cases"] for row in concentration["treated_cases_by_country"]}
    assert by_country[top] == max(by_country.values())
    assert (
        concentration["treated_cases_excluding_top"] + by_country[top]
        == pytest.approx(concentration["treated_cases_total"])
    )

    for row in concentration["treated_cases_by_event_time"]:
        assert row["top_country_cases"] <= row["cases"] + 1e-9
        assert row["countries"] <= row["units"]


def test_case_concentration_detects_a_planted_single_country_dominance():
    """A panel where one country carries the cases must report it as such.

    This is the guard that matters: the quantity exists to catch concentration, so
    a test that only checks arithmetic on a balanced panel would pass while the
    function measured the wrong thing.
    """
    panel = synthetic_panel(effect=-0.30).copy()
    treated = panel["treated"] == 1
    dominant = sorted(panel.loc[treated, "iso3"].unique())[0]
    panel.loc[treated & (panel["iso3"] == dominant), "cases"] *= 500

    concentration = A.treated_case_concentration(panel)
    assert concentration["treated_cases_top_country"] == dominant
    assert concentration["treated_cases_top_country_share"] > 0.9
    assert (
        concentration["treated_cases_median_per_cell_excluding_top"]
        < concentration["treated_cases_median_per_cell"]
    )


def test_poisson_pre_trend_uses_a_score_bootstrap_that_keeps_its_power(monkeypatch):
    """The count engine gets a score bootstrap, not a resampled outcome.

    A wild bootstrap would have to form ``fitted + sign * residual``, which can
    go negative, and a Poisson likelihood needs a non-negative outcome. The score
    bootstrap imposes the null once and reweights cluster score contributions.
    """
    monkeypatch.setattr(A, "BOOTSTRAP_REPLICATIONS", 399)

    flat = A.sun_abraham(
        synthetic_panel(pre_trend=0.0), 0, window=(-8, 6), engine="poisson"
    )
    block = flat["joint_pre_trend_bootstrap"]
    assert "score" in block["method"]
    assert block["restricted_fit_converged"] is True
    assert block["bootstrap_p_value"] > 0.05

    # Size is bought only if power survives it.
    sloped = A.sun_abraham(
        synthetic_panel(pre_trend=0.02), 0, window=(-8, 6), engine="poisson"
    )
    assert sloped["joint_pre_trend_bootstrap"]["bootstrap_p_value"] < 0.05


def test_each_engine_is_routed_to_its_own_bootstrap(monkeypatch):
    monkeypatch.setattr(A, "BOOTSTRAP_REPLICATIONS", 199)
    frame = synthetic_panel(pre_trend=0.0)
    methods = {
        engine: A.sun_abraham(frame, 0, window=(-8, 6), engine=engine)[
            "joint_pre_trend_bootstrap"
        ]["method"]
        for engine in ("ols", "poisson")
    }
    assert "score" not in methods["ols"]
    assert "score" in methods["poisson"]


def test_asymptotic_pre_trend_test_is_the_oversized_one(monkeypatch):
    """Locks in the reason the decision rule does not read the asymptotic p-value.

    Kept small (6 draws) so the suite stays quick; it is a regression guard on
    the calibration gap, not a precise size estimate.
    """
    monkeypatch.setattr(A, "BOOTSTRAP_REPLICATIONS", 399)

    asymptotic, bootstrap = [], []
    for seed in range(6):
        result = A.sun_abraham(
            synthetic_panel(pre_trend=0.0, seed=seed), 0, window=(-8, 6)
        )
        asymptotic.append(result["joint_pre_trend_test"]["p_value"])
        bootstrap.append(result["joint_pre_trend_bootstrap"]["bootstrap_p_value"])

    # Under a parallel DGP the bootstrap p-values should look roughly uniform,
    # so their median sits well away from zero; the asymptotic ones pile up low.
    assert np.median(bootstrap) > np.median(asymptotic)
    assert np.median(bootstrap) > 0.05


def test_anticipation_moves_the_base_period_and_keeps_leads_estimated():
    frame = synthetic_panel(effect=-0.30)
    result = A.sun_abraham(frame, anticipation=4, window=(-8, 6))
    assert result["reference_event_time"] == -5
    estimated = {row["event_time"] for row in result["event_study"]}
    assert -5 not in estimated, "the base period must not be estimated"
    assert {-4, -3, -2, -1}.issubset(estimated), "leads must remain estimated"


def test_event_time_zero_effect_is_detected_by_the_latency_rule():
    from estimation_report import evaluate_decision_rules

    frame = synthetic_panel(effect=-0.30)
    report = {
        "sample": {"treated_countries": 23, "treated_units": 29},
        "specifications": {
            "sun_abraham_log_rate": A.sun_abraham(frame, 0, window=(-8, 6)),
            "sun_abraham_poisson_count": A.sun_abraham(
                frame, 0, window=(-8, 6), engine="poisson"
            ),
            "callaway_santanna_primary": A.callaway_santanna(frame, 0),
        },
    }
    decisions = evaluate_decision_rules(report)
    latency = next(
        check for check in decisions["checks"]
        if check["rule"].startswith("no effect before plausible latency")
    )
    # The synthetic effect switches on exactly at event time 0, so the rule
    # must fire: it is a detector of "too early", not of "no effect".
    assert latency["passed"] is False


def test_twfe_is_biased_where_the_group_time_estimator_is_not():
    """Heterogeneous effects by cohort: TWFE should miss, Sun-Abraham should not."""
    rng = np.random.default_rng(0)
    years = np.arange(1995, 2024)
    rows = []
    # Later cohorts get much larger effects, which is what breaks pooled TWFE.
    for country, (cohort, effect) in {
        "AAA": (2003, -0.05), "BBB": (2011, -0.40), "CCC": (2017, -0.80),
    }.items():
        level = rng.normal(1.5, 0.2)
        for year in years:
            event_time = year - cohort
            rows.append({
                "iso3": country, "age_group": "20-24", "year": year,
                "treatment_cohort_year": cohort, "ever_treated": 1,
                "treated": int(event_time >= 0), "event_time": float(event_time),
                "log_rate": level + (effect if event_time >= 0 else 0.0)
                + rng.normal(scale=0.005),
            })
    for index in range(30):
        level = rng.normal(1.5, 0.2)
        for year in years:
            rows.append({
                "iso3": f"C{index:02d}", "age_group": "20-24", "year": year,
                "treatment_cohort_year": 0, "ever_treated": 0, "treated": 0,
                "event_time": np.nan,
                "log_rate": level + rng.normal(scale=0.005),
            })
    frame = pd.DataFrame(rows)
    frame["unit_id"] = frame["iso3"] + "|" + frame["age_group"]
    frame["cohort"] = frame["treatment_cohort_year"].replace(0, np.nan)
    frame["age_year"] = frame["age_group"] + ":" + frame["year"].astype(str)

    naive = A.twfe_comparison(frame)["estimate"]["estimate"]
    robust = A.sun_abraham(frame, 0, window=(-6, 6))["overall_post_att"]
    # Every true effect is at least -0.05 and the average is well below zero;
    # the pooled estimator is pulled toward zero by already-treated controls.
    assert robust < naive
