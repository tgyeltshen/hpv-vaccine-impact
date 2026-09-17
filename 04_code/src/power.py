"""How large an effect this design could detect, and how large one should be.

A feasibility paper has to answer two questions that the estimates alone do not.
First, what is the smallest effect the available information could distinguish from
zero. Second, how that compares with the effect the biology implies — because
"we found nothing" means something different when the design could see a 5%
reduction than when it could only see a 40% one.

The variance bound here is deliberately optimistic. It counts Poisson information
in incident cases and ignores clustering entirely, so the minimum detectable effect
it returns is the best the design could possibly do. Comparing it against the
standard errors the fitted models actually produce gives the design effect: the
factor by which clustering and case concentration inflate the achievable variance.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Two-sided 5% significance, 80% power.
Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143


def log_rate_variance(
    treated_post: float,
    treated_pre: float,
    control_post: float,
    control_pre: float,
) -> float:
    """Variance of a log rate ratio in a 2x2 count difference-in-differences.

    Each cell contributes ``1/D`` where ``D`` is its expected number of events, the
    standard Poisson information result. Cells with no events contribute infinite
    variance, which is the correct answer rather than an error.
    """
    counts = (treated_post, treated_pre, control_post, control_pre)
    if any(count <= 0 for count in counts):
        return float("inf")
    return float(sum(1.0 / count for count in counts))


def minimum_detectable_effect(
    standard_error: float, *, z_alpha: float = Z_ALPHA, z_power: float = Z_POWER
) -> dict[str, float]:
    """Smallest log rate ratio distinguishable from zero at the stated power."""
    if not np.isfinite(standard_error) or standard_error <= 0:
        return {"log_scale": float("inf"), "irr": 0.0, "reduction_pct": 100.0}
    log_scale = (z_alpha + z_power) * standard_error
    return {
        "log_scale": float(log_scale),
        "irr": float(np.exp(-log_scale)),
        "reduction_pct": float(100 * (1 - np.exp(-log_scale))),
    }


def design_effect(empirical_se: float, analytic_se: float) -> float:
    """Ratio of the achieved standard error to the information bound.

    A value of 1 means the design extracts all the information the case counts
    contain. Values above 1 measure what clustering and concentration cost.
    """
    if analytic_se <= 0 or not np.isfinite(empirical_se):
        return float("nan")
    return float(empirical_se / analytic_se)


def effective_cases(total_cases: float, design_effect_value: float) -> float:
    """Cases an unclustered design would need to match the achieved precision."""
    if not np.isfinite(design_effect_value) or design_effect_value <= 0:
        return float("nan")
    return float(total_cases / design_effect_value ** 2)


def expected_itt_reduction(
    coverage: float, efficacy: float, eligible_fraction: float = 1.0
) -> float:
    """Band-level intention-to-treat reduction implied by coverage and efficacy.

    An age band in which a fraction ``eligible_fraction`` of women were offered a
    vaccine, taken up by ``coverage`` of them, that prevents ``efficacy`` of cases
    among those vaccinated, has its incidence reduced by the product. This is an
    assumption-driven benchmark, not an estimate; the parameters are the reader's
    to vary.
    """
    for name, value in (("coverage", coverage), ("efficacy", efficacy),
                        ("eligible_fraction", eligible_fraction)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a proportion")
    return float(coverage * efficacy * eligible_fraction)


def case_counts_by_event_time(
    frame: pd.DataFrame, anticipation: int
) -> pd.DataFrame:
    """Treated and control event counts backing each event-time estimate."""
    base_period = -(anticipation + 1)
    treated_units = set(frame.loc[frame["ever_treated"] == 1, "unit_id"])

    is_base = (frame["unit_id"].isin(treated_units)) & (frame["event_time"] == base_period)
    treated_pre = float(frame.loc[is_base, "cases"].sum())

    controls = frame.loc[frame["ever_treated"] != 1]
    control_pre = float(controls.loc[controls["year"] <= frame.loc[is_base, "year"].max(),
                                     "cases"].sum())

    records = []
    treated = frame.loc[frame["treated"] == 1]
    for event_time, group in treated.groupby("event_time"):
        years = group["year"].unique()
        control_post = float(controls.loc[controls["year"].isin(years), "cases"].sum())
        records.append({
            "event_time": int(event_time),
            "units": int(group["unit_id"].nunique()),
            "treated_post_cases": float(group["cases"].sum()),
            "treated_pre_cases": treated_pre,
            "control_post_cases": control_post,
            "control_pre_cases": control_pre,
        })
    table = pd.DataFrame.from_records(records)
    table["analytic_variance"] = [
        log_rate_variance(row.treated_post_cases, row.treated_pre_cases,
                          row.control_post_cases, row.control_pre_cases)
        for row in table.itertuples()
    ]
    table["analytic_se"] = np.sqrt(table["analytic_variance"])
    table["analytic_mde_pct"] = [
        minimum_detectable_effect(se)["reduction_pct"] for se in table["analytic_se"]
    ]
    return table


def power_summary(
    frame: pd.DataFrame,
    anticipation: int,
    empirical_ses: Mapping[int, float],
    overall_se: float,
    overall_cases: float,
) -> dict[str, object]:
    """Information bound, realised precision, and the gap between them."""
    table = case_counts_by_event_time(frame, anticipation)
    table["empirical_se"] = [
        empirical_ses.get(int(event_time), float("nan"))
        for event_time in table["event_time"]
    ]
    table["design_effect"] = [
        design_effect(row.empirical_se, row.analytic_se) for row in table.itertuples()
    ]
    table["empirical_mde_pct"] = [
        minimum_detectable_effect(se)["reduction_pct"] if np.isfinite(se) else float("nan")
        for se in table["empirical_se"]
    ]

    treated_pre = float(table["treated_pre_cases"].iloc[0]) if len(table) else 0.0
    control_pre = float(table["control_pre_cases"].iloc[0]) if len(table) else 0.0
    control_post = float(table["control_post_cases"].max()) if len(table) else 0.0
    overall_variance = log_rate_variance(
        overall_cases, treated_pre, control_post, control_pre
    )
    overall_analytic_se = float(np.sqrt(overall_variance))
    overall_design_effect = design_effect(overall_se, overall_analytic_se)

    return {
        "by_event_time": table.to_dict(orient="records"),
        "overall": {
            "treated_cases": overall_cases,
            "analytic_se": overall_analytic_se,
            "analytic_mde": minimum_detectable_effect(overall_analytic_se),
            "empirical_se": float(overall_se),
            "empirical_mde": minimum_detectable_effect(overall_se),
            "design_effect": overall_design_effect,
            "effective_cases": effective_cases(overall_cases, overall_design_effect),
        },
        "expected_reduction_grid": [
            {
                "coverage": coverage,
                "efficacy": efficacy,
                "reduction_pct": 100 * expected_itt_reduction(coverage, efficacy),
            }
            for coverage in (0.4, 0.6, 0.8)
            for efficacy in (0.5, 0.7, 0.9)
        ],
    }
