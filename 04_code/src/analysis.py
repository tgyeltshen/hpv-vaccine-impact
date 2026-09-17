"""Staggered-adoption effect estimation for Project 1.

Implements the statistical analysis plan: a group-time estimator robust to
staggered timing and heterogeneous effects (Callaway-Sant'Anna), the
Sun-Abraham interaction-weighted event study, the protocol's Poisson
pseudo-maximum-likelihood count model with a population offset, and a
deliberately labelled two-way fixed-effect comparison.

Design facts that shape every specification here, all established in the
ingestion stage rather than assumed:

* The unit of adoption is country x five-year age band, because a birth cohort
  enters each band in a different calendar year. Treatment is absorbing.
* A five-year age band takes five years to be fully replaced by eligible birth
  cohorts, so event times -4 to -1 are *partially* eligible by construction and
  the ingestion stage labels them ambiguous. They are handled as an anticipation
  window, which moves the comparison base period back to ``-(anticipation+1)``
  and keeps those periods as estimated leads rather than as controls. The
  window length is varied as a sensitivity because it is a design choice.
* Standard errors cluster at country, not at unit: a country contributes up to
  four age bands whose outcomes are modelled jointly by GBD.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from panel_models import (
    FitResult,
    absorbed_ols,
    poisson_fe,
    poisson_score_bootstrap_wald,
    wald_test,
    wild_cluster_bootstrap_wald,
)

# Multiplier bootstrap used by ``differences`` for the Callaway-Sant'Anna
# aggregations.
BOOT_ITERATIONS = 999
# Replications for the restricted wild cluster bootstrap applied to the linear
# event study's joint pre-trend test. Kept separate from BOOT_ITERATIONS: this
# one is our own test, and 1999 makes (B + 1) * 0.05 an integer so the p-value
# has no granularity artefact at the conventional level.
BOOTSTRAP_REPLICATIONS = 1999
RANDOM_STATE = 20260812
ALPHA = 0.05

# Anticipation windows in event-time periods. Four is the primary: it is the
# width of the partial-overlap window that five-year age bands create
# mechanically. Zero and eight bracket it.
ANTICIPATION_GRID = (0, 4, 8)
PRIMARY_ANTICIPATION = 4

# Maximum age at which a birth cohort may first have been reached for its cell to
# count as treated under the prophylactic-plausibility restriction (D4). 15 is the
# primary: it is around the upper end of routine pre-debut schedules. 18 is the
# sensitivity, admitting late-adolescent catch-up.
PROPHYLACTIC_AGE_GRID = (15, 18)

# Falsification tests. The in-time leads are how far back a pseudo onset is
# placed; 8 is long enough to sit clear of the anticipation window. The adult
# threshold defines the negative-control exposure: cohorts first reached at or
# after this age, for whom a prophylactic vaccine cannot plausibly act.
PLACEBO_LEAD_GRID = (8, 12)
ADULT_ELIGIBILITY_MIN_AGE = 20

# Covariates offered to the doubly-robust specification. Only fields that vary
# over time and are present for most country-years qualify; the ingestion stage
# records both properties and the check is enforced in code below.
CANDIDATE_COVARIATES = (
    "log_gdp_per_capita_ppp",
    "urban_population_pct",
    "secondary_enrolment_gross_pct",
    "hiv_prevalence_15_49_pct",
    "uhc_service_coverage_index",
)
MIN_COVARIATE_COMPLETENESS = 0.90


# ---------------------------------------------------------------------------
# estimation frame
# ---------------------------------------------------------------------------


@dataclass
class EstimationFrame:
    data: pd.DataFrame
    covariates: list[str]
    notes: dict[str, object]


def _reassign_treatment(
    frame: pd.DataFrame, eligible: pd.Series
) -> pd.DataFrame:
    """Rebuild the absorbing treatment columns from a per-cell eligibility mask.

    Reassignment rather than filtering, because dropping rows would leave a
    unit's ``treatment_cohort_year`` pointing at a year that no longer qualifies
    and every event time downstream would be measured from the wrong origin.
    Units with no qualifying year become never-treated and serve as controls.
    """
    data = frame.copy()
    first = data.loc[eligible].groupby(["iso3", "age_group"])["year"].min()
    keys = pd.MultiIndex.from_arrays([data["iso3"], data["age_group"]])
    cohort_year = pd.Series(first.reindex(keys).to_numpy(), index=data.index)

    data["treatment_cohort_year"] = cohort_year.fillna(0).astype(int)
    data["cohort"] = cohort_year
    data["ever_treated"] = cohort_year.notna().astype(int)
    data["treated"] = (cohort_year.notna() & (data["year"] >= cohort_year)).astype(int)
    data["event_time"] = np.where(
        cohort_year.notna(), data["year"] - cohort_year, np.nan
    )
    return data


def placebo_in_time(frame: pd.DataFrame, lead_years: int) -> pd.DataFrame:
    """Move each treated unit's onset earlier and drop its real post-period.

    A falsification test for the trend that the event-time-0 result might be
    picking up. Every genuinely post-eligibility observation is removed, so
    nothing the programme could have caused remains in the panel; the pseudo
    onset then sits ``lead_years`` before the real one. An estimated "effect"
    here is a pre-existing divergence between adopters and non-adopters, because
    there is no longer any treatment for it to be.
    """
    data = frame.copy()
    cohort = data["cohort"]
    data = data.loc[cohort.isna() | (data["year"] < cohort)].copy()

    pseudo = data["cohort"] - lead_years
    data["treatment_cohort_year"] = pseudo.fillna(0).astype(int)
    data["cohort"] = pseudo
    data["ever_treated"] = pseudo.notna().astype(int)
    data["treated"] = (pseudo.notna() & (data["year"] >= pseudo)).astype(int)
    data["event_time"] = np.where(pseudo.notna(), data["year"] - pseudo, np.nan)
    return data


def restrict_to_adult_eligibility(
    frame: pd.DataFrame, min_age_at_eligibility: int
) -> pd.DataFrame:
    """Negative-control exposure: only cohorts first reached in adulthood.

    The mirror image of :func:`restrict_to_prophylactic_age`. A prophylactic
    vaccine given to a cohort already past sexual debut cannot plausibly change
    that cohort's invasive cervical cancer incidence, so an effect estimated on
    these cells alone measures whatever confounding accompanies programme
    adoption. If it is as large as the effect among cohorts reached in childhood,
    the association is not vaccine-mediated.
    """
    eligible = (frame["primary_treated_eligible"] == 1) & (
        pd.to_numeric(frame["age_at_eligibility_max"], errors="coerce")
        >= min_age_at_eligibility
    )
    return _reassign_treatment(frame, eligible)


def restrict_to_prophylactic_age(
    frame: pd.DataFrame, max_age_at_eligibility: int
) -> pd.DataFrame:
    """Reassign treatment, admitting only cohorts reached young enough to benefit.

    HPV vaccine is prophylactic. A birth cohort first reached at 25 cannot have
    its invasive cervical cancer risk altered the way one reached at 12 can, so a
    cell whose band was only ever targeted in adulthood carries an eligibility
    label that no biological mechanism connects to the outcome. Those cells sit at
    event times 0 and above, which is exactly where an implausible immediate
    effect would appear.

    This has to reassign rather than filter. Dropping rows would leave a unit's
    ``treatment_cohort_year`` pointing at a year that no longer qualifies, so
    every event time downstream would be measured from the wrong origin. Units
    with no qualifying year become never-treated and serve as controls, which is
    the right place for them: nothing documented makes them exposed.
    """
    eligible = (frame["primary_treated_eligible"] == 1) & (
        pd.to_numeric(frame["age_at_eligibility_max"], errors="coerce")
        <= max_age_at_eligibility
    )
    return _reassign_treatment(frame, eligible)


def treated_cell_counts(frame: pd.DataFrame) -> dict[str, object]:
    """Both treated-cell counts, plus the cells they disagree about.

    Two definitions are in play and conflating them invites the reader to think
    one of them is wrong. ``treated`` is the absorbing indicator the design runs
    on: programme eligibility cannot be withdrawn from a birth cohort once it has
    been targeted, so it is 1 from the unit's first treated year onward.
    ``treated_documented`` is the per-cell flag, 1 only where the whole age band
    falls inside that year's documented target ages. Cells counted by the first
    and not the second are listed rather than smoothed, and by construction the
    absorbing count is the documented count plus that list.
    """
    absorbing = frame["treated"] == 1
    documented = frame["analysis_status"] == "treated_documented"
    disagreeing = absorbing & ~documented
    return {
        "treated_cell_count": int(absorbing.sum()),
        "treated_cells_documented": int(documented.sum()),
        "treated_cells_absorbing_only": int(disagreeing.sum()),
        "treated_cells_absorbing_only_detail": [
            {
                "iso3": row["iso3"], "age_group": row["age_group"],
                "year": int(row["year"]), "event_time": int(row["event_time"]),
                "analysis_status": row["analysis_status"],
            }
            for _, row in frame.loc[disagreeing].iterrows()
        ],
    }


def treated_case_concentration(frame: pd.DataFrame) -> dict[str, object]:
    """Where the information actually is.

    Standard errors on this panel track incident case mass, not the number of
    contributing countries, so a reader watching confidence intervals alone gets
    no warning when the estimate narrows onto a single country. That has to be
    reported directly rather than inferred. This returns the share of treated-cell
    cases held by each country, and how the leading country's share moves as
    post-treatment support collapses.

    The leading country is identified from the data rather than named here, so
    that the check keeps working if the exposure assignment changes.
    """
    treated = frame.loc[frame["treated"] == 1]
    total = float(treated["cases"].sum())
    by_country = treated.groupby("iso3")["cases"].sum().sort_values(ascending=False)
    top_country = str(by_country.index[0])
    others = treated.loc[treated["iso3"] != top_country]

    by_event_time = []
    for event_time, group in treated.groupby("event_time"):
        cases = float(group["cases"].sum())
        top = float(group.loc[group["iso3"] == top_country, "cases"].sum())
        by_event_time.append({
            "event_time": int(event_time),
            "units": int(group["unit_id"].nunique()),
            "countries": int(group["iso3"].nunique()),
            "cases": cases,
            "top_country_cases": top,
            "top_country_share": (top / cases) if cases else None,
        })

    return {
        "treated_cases_total": total,
        "treated_cases_top_country": top_country,
        "treated_cases_top_country_share": float(by_country.iloc[0] / total),
        "treated_cases_by_country": [
            {"iso3": str(iso3), "cases": float(value), "share": float(value / total)}
            for iso3, value in by_country.items()
        ],
        "treated_cases_median_per_cell": float(treated["cases"].median()),
        "treated_cases_excluding_top": float(others["cases"].sum()),
        "treated_cells_excluding_top": int(len(others)),
        "treated_cases_median_per_cell_excluding_top": float(others["cases"].median()),
        "treated_cases_by_event_time": by_event_time,
    }


def load_estimation_frame(root: Path) -> EstimationFrame:
    path = root / "03_processed_data" / "analytic_dataset.csv"
    frame = pd.read_csv(path)

    frame["log_rate"] = np.log(frame["rate_per_100k"])
    # GBD publishes cases and rates on its own population estimates. Dividing
    # them recovers that denominator exactly, which keeps the count model
    # internally consistent; the UN WPP denominator is retained separately and
    # the disagreement between the two is reported as a data-quality check.
    frame["gbd_population"] = frame["cases"] / frame["rate_per_100k"] * 1e5
    frame["log_gbd_population"] = np.log(frame["gbd_population"])
    frame["log_wpp_population"] = np.log(frame["female_population"])
    frame["denominator_rel_diff"] = (
        (frame["gbd_population"] - frame["female_population"]).abs()
        / frame["female_population"]
    )

    frame["cohort"] = frame["treatment_cohort_year"].replace(0, np.nan)
    frame["age_year"] = frame["age_group"] + ":" + frame["year"].astype(str)
    with np.errstate(divide="ignore"):
        frame["log_gdp_per_capita_ppp"] = np.log(frame["gdp_per_capita_ppp"])
    frame.loc[~np.isfinite(frame["log_gdp_per_capita_ppp"]), "log_gdp_per_capita_ppp"] = np.nan

    # Protocol section "Uncertainty", option 2: the modelled outcome carries a
    # published interval but no draws, so a log-scale standard error is
    # approximated from the bounds. This is an approximation and is labelled as
    # one wherever it is used.
    frame["log_rate_se_approx"] = (
        np.log(frame["rate_per_100k_upper"]) - np.log(frame["rate_per_100k_lower"])
    ) / (2 * 1.959963984540054)

    completeness = {
        name: float(frame[name].notna().mean())
        for name in CANDIDATE_COVARIATES
        if name in frame.columns
    }
    usable = [
        name for name, share in completeness.items()
        if share >= MIN_COVARIATE_COMPLETENESS
    ]
    notes = {
        "rows": int(len(frame)),
        "units": int(frame["unit_id"].nunique()),
        "countries": int(frame["iso3"].nunique()),
        "years": [int(frame["year"].min()), int(frame["year"].max())],
        "treated_units": int(frame.loc[frame["ever_treated"] == 1, "unit_id"].nunique()),
        "treated_countries": int(frame.loc[frame["ever_treated"] == 1, "iso3"].nunique()),
        **treated_cell_counts(frame),
        **treated_case_concentration(frame),
        "covariate_completeness": completeness,
        "covariates_used_in_doubly_robust": usable,
        "covariates_rejected_for_incompleteness": {
            name: share for name, share in completeness.items() if name not in usable
        },
        "gbd_vs_wpp_denominator_median_rel_diff": float(
            frame["denominator_rel_diff"].median()
        ),
        "gbd_vs_wpp_denominator_p99_rel_diff": float(
            frame["denominator_rel_diff"].quantile(0.99)
        ),
        "outcome_uncertainty_median_log_se": float(frame["log_rate_se_approx"].median()),
    }
    return EstimationFrame(data=frame, covariates=usable, notes=notes)


def support_table(frame: pd.DataFrame, anticipation: int) -> list[dict[str, object]]:
    """Units and countries observed at each event time, with their status mix."""
    treated = frame[frame["ever_treated"] == 1].copy()
    treated["event_time"] = treated["event_time"].astype(int)
    rows = []
    for event_time, block in treated.groupby("event_time"):
        rows.append(
            {
                "event_time": int(event_time),
                "units": int(block["unit_id"].nunique()),
                "countries": int(block["iso3"].nunique()),
                "cohorts": int(block["treatment_cohort_year"].nunique()),
                "ambiguous_rows": int(
                    block["analysis_status"].str.startswith("ambiguous").sum()
                ),
                "is_base_period": bool(event_time == -(anticipation + 1)),
                "in_anticipation_window": bool(-anticipation <= event_time <= -1),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Callaway-Sant'Anna
# ---------------------------------------------------------------------------


def _attgt(
    frame: pd.DataFrame,
    anticipation: int,
    control_group: str,
    covariates: list[str],
    est_method: str,
    base_period: str = "universal",
):
    from differences import ATTgt

    columns = ["unit_id", "year", "log_rate", "cohort", "iso3"] + covariates
    panel = frame[columns].set_index(["unit_id", "year"]).sort_index()
    formula = "log_rate"
    if covariates:
        formula = "log_rate ~ " + " + ".join(covariates)

    # A universal base period holds every lead and lag against the same
    # reference, ``-(anticipation+1)``. The library default ('varying') compares
    # each pre-period with the one before it, which produces year-on-year
    # changes that cannot be read on the same axis as the Sun-Abraham leads and
    # would understate a smooth pre-trend. 'varying' is kept as a sensitivity.
    model = ATTgt(
        data=panel,
        cohort_column="cohort",
        anticipation=anticipation,
        base_period=base_period,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            formula=formula,
            control_group=control_group,
            est_method=est_method,
            progress_bar=False,
        )
    # differences 0.3.0 cannot accept ``cluster_var`` in ``fit`` (it nests the
    # argument into a column selector and raises). The aggregation entry point
    # accepts it correctly, so the cluster column is attached to the fitted
    # design here. ``model.data`` carries the same encoded panel index as
    # ``model._data_matrix``, so this is an index-aligned assignment rather
    # than a positional one.
    model._data_matrix["iso3"] = model.data["iso3"]
    if model._data_matrix["iso3"].isna().any():
        raise RuntimeError("cluster column did not align with the fitted design")
    return model


def _aggregation_frame(model, kind: str, overall: bool = False) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.aggregate(
            kind,
            overall=overall,
            cluster_var=["iso3"],
            boot_iterations=BOOT_ITERATIONS,
            random_state=RANDOM_STATE,
            alpha=ALPHA,
        )
    frame = result.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple) else str(column)
        for column in frame.columns
    ]
    return frame.reset_index()


def _tidy_aggregation(frame: pd.DataFrame, index_name: str) -> list[dict[str, object]]:
    renames = {}
    for column in frame.columns:
        low = column.lower()
        if low.endswith("att") or low == "att":
            renames[column] = "estimate"
        elif "std_error" in low:
            renames[column] = "std_error"
        elif low.endswith("lower"):
            renames[column] = "conf_low"
        elif low.endswith("upper"):
            renames[column] = "conf_high"
    frame = frame.rename(columns=renames)
    first = frame.columns[0]
    frame = frame.rename(columns={first: index_name})
    keep = [index_name, "estimate", "std_error", "conf_low", "conf_high"]
    keep = [column for column in keep if column in frame.columns]
    records = frame[keep].to_dict("records")
    for record in records:
        estimate = record.get("estimate")
        if estimate is not None and np.isfinite(estimate):
            record["irr"] = float(np.exp(estimate))
            if record.get("conf_low") is not None and np.isfinite(record["conf_low"]):
                record["irr_conf_low"] = float(np.exp(record["conf_low"]))
                record["irr_conf_high"] = float(np.exp(record["conf_high"]))
    return records


def callaway_santanna(
    frame: pd.DataFrame,
    anticipation: int,
    control_group: str = "never_treated",
    covariates: list[str] | None = None,
    est_method: str = "reg",
    base_period: str = "universal",
) -> dict[str, object]:
    covariates = covariates or []
    model = _attgt(
        frame, anticipation, control_group, covariates, est_method, base_period
    )

    event = _tidy_aggregation(_aggregation_frame(model, "event"), "event_time")
    simple = _tidy_aggregation(_aggregation_frame(model, "simple"), "aggregation")
    cohort = _tidy_aggregation(_aggregation_frame(model, "cohort"), "cohort")

    # differences 0.3.0 ships a broken ``wald_pre_test`` (it calls the
    # ``results`` property with an argument). The joint pre-trend inference the
    # protocol requires is produced instead by the Sun-Abraham aggregation,
    # which tests the same leads on the same panel with a covariance we control;
    # the per-lead Callaway-Sant'Anna estimates below are still reported.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre_test = model.wald_pre_test()
    except Exception as error:
        pre_test = {
            "available": False,
            "reason": f"differences 0.3.0 defect: {error}",
            "joint_test_reported_by": "sun_abraham_log_rate.joint_pre_trend_test",
        }

    group_time = model.results().reset_index()

    post = [row for row in event if row["event_time"] >= 0]
    pre = [row for row in event if row["event_time"] < 0]
    return {
        "anticipation": anticipation,
        "control_group": control_group,
        "est_method": est_method,
        "base_period": base_period,
        "covariates": covariates,
        "n_entities_used": int(model._data_matrix.index.get_level_values(0).nunique()),
        "n_rows_used": int(len(model._data_matrix)),
        "event_study": event,
        "overall_att": simple,
        "by_cohort": cohort,
        "wald_pre_test": _wald_to_dict(pre_test),
        "n_pre_periods_estimated": len(pre),
        "n_post_periods_estimated": len(post),
        "group_time_rows": len(group_time),
    }


def _wald_to_dict(pre_test) -> dict[str, object]:
    if pre_test is None:
        return {}
    if isinstance(pre_test, pd.DataFrame):
        record = pre_test.reset_index().to_dict("records")
        return {"table": record}
    if isinstance(pre_test, dict):
        return {key: _to_native(value) for key, value in pre_test.items()}
    return {"value": _to_native(pre_test)}


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict("records")
    return value


# ---------------------------------------------------------------------------
# Sun-Abraham interaction-weighted estimator
# ---------------------------------------------------------------------------


def _event_window(frame: pd.DataFrame, low: int, high: int) -> pd.Series:
    event_time = frame["event_time"]
    binned = event_time.clip(lower=low, upper=high)
    return binned


def _pre_trend_bootstrap(
    fit: FitResult,
    engine: str,
    pre_terms: list[str],
    weight_matrix: np.ndarray,
    pre_mask: np.ndarray,
    replications: int | None = None,
) -> dict[str, object]:
    """Bootstrap the joint pre-trend test with the device each engine allows.

    Both engines test the same hypothesis on the same interaction-weighted
    event-time aggregates, and both impose the null by holding the lead
    interactions at zero. They differ only in what can be resampled: the linear
    engine refits a resampled outcome, while the Poisson engine reweights
    cluster score contributions, because a resampled Poisson outcome is not
    guaranteed to be non-negative.
    """
    if not pre_mask.any():
        return {
            "bootstrap_p_value": None,
            "reason": "no pre-period event times were estimated",
        }
    n_bootstrap = (
        BOOTSTRAP_REPLICATIONS if replications is None else int(replications)
    )
    if n_bootstrap <= 0:
        return {
            "bootstrap_p_value": None,
            "reason": "bootstrap not requested for this specification",
        }
    if engine == "ols":
        return wild_cluster_bootstrap_wald(
            fit, pre_terms, weights=weight_matrix[pre_mask],
            n_bootstrap=n_bootstrap, seed=RANDOM_STATE,
        )
    if engine == "poisson":
        return poisson_score_bootstrap_wald(
            fit, pre_terms, restriction=weight_matrix[pre_mask],
            n_bootstrap=n_bootstrap, seed=RANDOM_STATE,
        )
    return {
        "bootstrap_p_value": None,
        "reason": f"no bootstrap is implemented for engine {engine!r}",
    }


def sun_abraham(
    frame: pd.DataFrame,
    anticipation: int,
    window: tuple[int, int] = (-10, 6),
    min_units_per_cell: int = 1,
    engine: str = "ols",
    offset_column: str = "log_gbd_population",
    bootstrap_replications: int | None = None,
) -> dict[str, object]:
    """Interaction-weighted event study (Sun and Abraham).

    Cohort-specific relative-time effects are estimated in a saturated
    regression with unit and age-by-year fixed effects, then averaged across
    cohorts using each cohort's share of the units observed at that relative
    time. The averaging weights are treated as known, which is the usual
    implementation and understates uncertainty slightly.

    ``engine='ols'`` estimates on the log rate and weights every unit equally.
    ``engine='poisson'`` estimates the protocol's count model, which weights
    each observation by its fitted case mass; with this panel that is not a
    cosmetic difference, because two Brazilian units hold most of the treated
    case mass. Both are reported rather than one being chosen.
    """
    low, high = window
    reference = -(anticipation + 1)
    data = frame.copy()
    treated_mask = data["ever_treated"] == 1
    data["event_binned"] = np.where(
        treated_mask, _event_window(data, low, high), np.nan
    )

    cells = (
        data.loc[treated_mask, ["treatment_cohort_year", "event_binned", "unit_id"]]
        .dropna()
        .groupby(["treatment_cohort_year", "event_binned"])["unit_id"]
        .nunique()
    )
    usable_cells = [
        (int(g), int(e)) for (g, e), n in cells.items()
        if n >= min_units_per_cell and int(e) != reference
    ]
    if not usable_cells:
        raise ValueError("no cohort x relative-time cells available")

    columns, names = [], []
    for cohort, event_time in usable_cells:
        indicator = (
            treated_mask
            & (data["treatment_cohort_year"] == cohort)
            & (data["event_binned"] == event_time)
        ).astype(float)
        columns.append(indicator.to_numpy())
        names.append(f"g{cohort}:e{event_time}")
    design = np.column_stack(columns)

    if engine == "ols":
        fit = absorbed_ols(
            y=data["log_rate"].to_numpy(),
            X=design,
            names=names,
            absorb=[data["unit_id"].to_numpy(), data["age_year"].to_numpy()],
            cluster=data["iso3"].to_numpy(),
        )
    elif engine == "poisson":
        age_year = pd.get_dummies(data["age_year"], drop_first=True).to_numpy(dtype=float)
        fit = poisson_fe(
            y=data["cases"].to_numpy(),
            X=np.column_stack([design, age_year]),
            names=names + [f"ay{i}" for i in range(age_year.shape[1])],
            offset=data[offset_column].to_numpy(),
            absorb_unit=data["unit_id"].to_numpy(),
            cluster=data["iso3"].to_numpy(),
        )
    else:
        raise ValueError(f"unknown engine: {engine}")

    # Interaction weights: cohort shares among units observed at each relative
    # time, computed on the same sample that identifies the interactions.
    share_source = (
        data.loc[treated_mask, ["treatment_cohort_year", "event_binned", "unit_id"]]
        .dropna()
        .drop_duplicates()
    )
    index = {name: position for position, name in enumerate(fit.names)}
    aggregated = []
    weight_rows: list[np.ndarray] = []
    for event_time in sorted(share_source["event_binned"].unique()):
        event_time = int(event_time)
        if event_time == reference:
            continue
        block = share_source[share_source["event_binned"] == event_time]
        counts = block.groupby("treatment_cohort_year")["unit_id"].nunique()
        total = counts.sum()
        weights, positions = [], []
        for cohort, count in counts.items():
            key = f"g{int(cohort)}:e{event_time}"
            if key in index:
                weights.append(count / total)
                positions.append(index[key])
        if not positions:
            continue
        weights = np.asarray(weights)
        # Cohorts whose interaction was dropped as collinear leave the retained
        # weights not summing to one; renormalising would silently reassign
        # their weight, so the realised weight sum is reported instead.
        estimate = float(weights @ fit.coef[positions])
        variance = float(weights @ fit.vcov[np.ix_(positions, positions)] @ weights)
        error = float(np.sqrt(max(variance, 0.0)))
        row_weights = np.zeros(len(fit.names))
        row_weights[positions] = weights
        weight_rows.append(row_weights)
        aggregated.append(
            {
                "event_time": event_time,
                "estimate": estimate,
                "std_error": error,
                "conf_low": estimate - 1.959963984540054 * error,
                "conf_high": estimate + 1.959963984540054 * error,
                "irr": float(np.exp(estimate)),
                "cohorts_used": len(positions),
                "units": int(block["unit_id"].nunique()),
                "weight_sum_retained": float(weights.sum()),
            }
        )

    interactions = [name for name in fit.names if ":e" in name]
    post_terms = [name for name in interactions if int(name.split(":e")[1]) >= 0]
    pre_terms = [
        name for name in interactions if int(name.split(":e")[1]) < -anticipation
    ]

    # The joint tests that matter are on the interaction-weighted event-time
    # estimates, not on every cohort-specific interaction: with five singleton
    # cohorts the raw interaction covariance is near-singular, whereas the
    # aggregated one has at most a dozen rows.
    weight_matrix = np.vstack(weight_rows) if weight_rows else np.zeros((0, len(fit.names)))
    aggregate_vcov = weight_matrix @ fit.vcov @ weight_matrix.T
    aggregate_estimates = np.asarray([row["estimate"] for row in aggregated])
    event_times = np.asarray([row["event_time"] for row in aggregated])
    pre_mask = event_times < -anticipation
    post_mask = event_times >= 0
    post_rows = [row for row in aggregated if row["event_time"] >= 0]
    overall = overall_se = None
    if post_rows:
        positions = [
            position for position, row in enumerate(aggregated) if row["event_time"] >= 0
        ]
        weights = np.asarray([row["units"] for row in post_rows], dtype=float)
        weights /= weights.sum()
        overall = float(weights @ np.asarray([row["estimate"] for row in post_rows]))
        combined = weights @ np.vstack([weight_rows[position] for position in positions])
        overall_se = float(np.sqrt(max(float(combined @ fit.vcov @ combined), 0.0)))

    return {
        "engine": engine,
        "anticipation": anticipation,
        "reference_event_time": reference,
        "window": [low, high],
        "n_obs": fit.n_obs,
        "n_clusters": fit.n_clusters,
        "n_interactions": len(interactions),
        "dropped_collinear_terms": fit.diagnostics.get("dropped_collinear_terms", []),
        "event_study": aggregated,
        "overall_post_att": overall,
        "overall_post_std_error": overall_se,
        "overall_post_conf_low": (
            None if overall is None else overall - 1.959963984540054 * overall_se
        ),
        "overall_post_conf_high": (
            None if overall is None else overall + 1.959963984540054 * overall_se
        ),
        "overall_post_irr": None if overall is None else float(np.exp(overall)),
        "joint_pre_trend_test": wald_test(
            aggregate_estimates[pre_mask],
            aggregate_vcov[np.ix_(pre_mask, pre_mask)],
        ),
        "joint_pre_trend_bootstrap": _pre_trend_bootstrap(
            fit, engine, pre_terms, weight_matrix, pre_mask,
            replications=bootstrap_replications,
        ),
        "joint_post_test": wald_test(
            aggregate_estimates[post_mask],
            aggregate_vcov[np.ix_(post_mask, post_mask)],
        ),
        "joint_pre_trend_test_on_interactions": fit.wald(pre_terms),
        "joint_post_test_on_interactions": fit.wald(post_terms),
    }


# ---------------------------------------------------------------------------
# count model and two-way fixed effects
# ---------------------------------------------------------------------------


def _event_design(
    frame: pd.DataFrame, anticipation: int, window: tuple[int, int]
) -> tuple[np.ndarray, list[str]]:
    low, high = window
    reference = -(anticipation + 1)
    treated_mask = frame["ever_treated"] == 1
    binned = np.where(treated_mask, _event_window(frame, low, high), np.nan)
    values = sorted({int(v) for v in binned[~np.isnan(binned)]} - {reference})
    columns, names = [], []
    for value in values:
        columns.append(((~np.isnan(binned)) & (binned == value)).astype(float))
        names.append(f"e{value}")
    return np.column_stack(columns), names


def poisson_event_study(
    frame: pd.DataFrame,
    anticipation: int,
    window: tuple[int, int] = (-10, 6),
    offset_column: str = "log_gbd_population",
) -> dict[str, object]:
    """Protocol primary count model: Poisson PML with a population offset."""
    design, names = _event_design(frame, anticipation, window)
    fit = poisson_fe(
        y=frame["cases"].to_numpy(),
        X=design,
        names=names,
        offset=frame[offset_column].to_numpy(),
        absorb_unit=frame["unit_id"].to_numpy(),
        cluster=frame["iso3"].to_numpy(),
    )
    # Age-by-year effects enter as explicit dummies alongside the event terms so
    # that the profiled unit effect stays closed-form.
    age_year = pd.get_dummies(frame["age_year"], drop_first=True).to_numpy(dtype=float)
    full_design = np.column_stack([design, age_year])
    full_names = names + [f"ay{i}" for i in range(age_year.shape[1])]
    fit_full = poisson_fe(
        y=frame["cases"].to_numpy(),
        X=full_design,
        names=full_names,
        offset=frame[offset_column].to_numpy(),
        absorb_unit=frame["unit_id"].to_numpy(),
        cluster=frame["iso3"].to_numpy(),
    )

    rows = [row for row in fit_full.as_table(ALPHA) if row["term"].startswith("e")]
    for row in rows:
        row["event_time"] = int(row["term"][1:])
        row["irr"] = float(np.exp(row["estimate"]))
        row["irr_conf_low"] = float(np.exp(row["conf_low"]))
        row["irr_conf_high"] = float(np.exp(row["conf_high"]))

    static_column = (
        (frame["ever_treated"] == 1) & (frame["event_time"].fillna(-999) >= 0)
    ).astype(float).to_numpy()[:, None]
    anticipation_column = (
        (frame["ever_treated"] == 1)
        & (frame["event_time"].fillna(-999) >= -anticipation)
        & (frame["event_time"].fillna(-999) < 0)
    ).astype(float).to_numpy()[:, None]
    static_design = np.column_stack(
        [static_column, anticipation_column, age_year]
    )
    static_names = ["treated_post", "anticipation_window"] + [
        f"ay{i}" for i in range(age_year.shape[1])
    ]
    static_fit = poisson_fe(
        y=frame["cases"].to_numpy(),
        X=static_design,
        names=static_names,
        offset=frame[offset_column].to_numpy(),
        absorb_unit=frame["unit_id"].to_numpy(),
        cluster=frame["iso3"].to_numpy(),
    )
    static_rows = [
        row for row in static_fit.as_table(ALPHA)
        if row["term"] in {"treated_post", "anticipation_window"}
    ]
    for row in static_rows:
        row["irr"] = float(np.exp(row["estimate"]))
        row["irr_conf_low"] = float(np.exp(row["conf_low"]))
        row["irr_conf_high"] = float(np.exp(row["conf_high"]))

    pre_terms = [
        name for name in names if int(name[1:]) < 0 and int(name[1:]) < -anticipation
    ]
    return {
        "anticipation": anticipation,
        "offset": offset_column,
        "converged": bool(fit_full.converged and static_fit.converged),
        "iterations": fit_full.iterations,
        "n_obs": fit_full.n_obs,
        "n_clusters": fit_full.n_clusters,
        "pearson_dispersion": fit_full.diagnostics["pearson_dispersion"],
        "event_study": rows,
        "static": static_rows,
        "joint_pre_trend_test": fit_full.wald(pre_terms),
        "without_age_year_effects_event_study": [
            {**row, "event_time": int(row["term"][1:])}
            for row in fit.as_table(ALPHA)
        ],
    }


def twfe_comparison(frame: pd.DataFrame) -> dict[str, object]:
    """Naive static two-way fixed effects. Reported only as a labelled contrast."""
    design = frame["treated"].to_numpy(dtype=float)[:, None]
    fit = absorbed_ols(
        y=frame["log_rate"].to_numpy(),
        X=design,
        names=["treated"],
        absorb=[frame["unit_id"].to_numpy(), frame["age_year"].to_numpy()],
        cluster=frame["iso3"].to_numpy(),
    )
    row = fit.as_table(ALPHA)[0]
    row["irr"] = float(np.exp(row["estimate"]))
    row["irr_conf_low"] = float(np.exp(row["conf_low"]))
    row["irr_conf_high"] = float(np.exp(row["conf_high"]))
    return {
        "label": "naive two-way fixed effects; not robust to staggered timing "
                 "or heterogeneous effects",
        "n_obs": fit.n_obs,
        "n_clusters": fit.n_clusters,
        "estimate": row,
    }


# ---------------------------------------------------------------------------
# influence and robustness
# ---------------------------------------------------------------------------


def leave_one_country_out(
    frame: pd.DataFrame, anticipation: int, control_group: str = "never_treated"
) -> list[dict[str, object]]:
    """Refit the overall ATT dropping each treated country in turn."""
    treated_countries = sorted(frame.loc[frame["ever_treated"] == 1, "iso3"].unique())
    results = []
    for country in treated_countries:
        subset = frame[frame["iso3"] != country]
        try:
            fitted = callaway_santanna(subset, anticipation, control_group)
            overall = fitted["overall_att"][0]
            results.append(
                {
                    "dropped_country": country,
                    "estimate": overall.get("estimate"),
                    "std_error": overall.get("std_error"),
                    "irr": overall.get("irr"),
                    "treated_units_remaining": int(
                        subset.loc[subset["ever_treated"] == 1, "unit_id"].nunique()
                    ),
                }
            )
        except Exception as error:  # a dropped country can empty a whole cohort
            results.append({"dropped_country": country, "error": str(error)})
    return results


def outcome_uncertainty_scenarios(
    frame: pd.DataFrame,
    anticipation: int,
    draws: int = 200,
    correlations: tuple[float, ...] = (0.0, 0.5, 0.9),
) -> list[dict[str, object]]:
    """Propagate the modelled-outcome interval into the overall ATT.

    GBD supplies an interval but no draws, so the log-scale standard error is
    approximated from the published bounds and resampled under explicitly
    varied within-unit temporal correlation. The protocol requires this to be
    labelled an approximation; it is not a substitute for draw-level data.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    base = frame.copy().sort_values(["unit_id", "year"]).reset_index(drop=True)
    unit_codes, _ = pd.factorize(base["unit_id"])
    sigma = base["log_rate_se_approx"].to_numpy()
    baseline_log = base["log_rate"].to_numpy()
    scenarios = []
    for rho in correlations:
        estimates = []
        for _ in range(draws):
            shocks = rng.standard_normal(len(base))
            if rho > 0:
                # AR(1) within unit; the first year of each unit starts the chain.
                correlated = np.empty_like(shocks)
                previous, previous_unit = 0.0, -1
                scale = np.sqrt(1 - rho ** 2)
                for position in range(len(shocks)):
                    if unit_codes[position] != previous_unit:
                        previous, previous_unit = shocks[position], unit_codes[position]
                    else:
                        previous = rho * previous + scale * shocks[position]
                    correlated[position] = previous
                shocks = correlated
            trial = base.copy()
            trial["log_rate"] = baseline_log + sigma * shocks
            try:
                fitted = callaway_santanna(trial, anticipation)
                estimates.append(fitted["overall_att"][0]["estimate"])
            except Exception:
                continue
        estimates = np.asarray([e for e in estimates if e is not None and np.isfinite(e)])
        if estimates.size == 0:
            scenarios.append({"within_unit_correlation": rho, "draws_completed": 0})
            continue
        scenarios.append(
            {
                "within_unit_correlation": rho,
                "draws_completed": int(estimates.size),
                "mean_att": float(estimates.mean()),
                "std_dev_att": float(estimates.std(ddof=1)),
                "percentile_2_5": float(np.percentile(estimates, 2.5)),
                "percentile_97_5": float(np.percentile(estimates, 97.5)),
                "mean_irr": float(np.exp(estimates.mean())),
            }
        )
    return scenarios
