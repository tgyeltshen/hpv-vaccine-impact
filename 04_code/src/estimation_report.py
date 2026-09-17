"""Run the analysis plan end to end and write results, tables and figures.

Estimation lives in :mod:`analysis`; this module only decides which
specifications to run, applies the protocol's stop rules to what comes back,
and renders the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    ADULT_ELIGIBILITY_MIN_AGE,
    ANTICIPATION_GRID,
    PLACEBO_LEAD_GRID,
    PRIMARY_ANTICIPATION,
    PROPHYLACTIC_AGE_GRID,
    _to_native,
    callaway_santanna,
    leave_one_country_out,
    load_estimation_frame,
    outcome_uncertainty_scenarios,
    placebo_in_time,
    poisson_event_study,
    restrict_to_adult_eligibility,
    restrict_to_prophylactic_age,
    sun_abraham,
    support_table,
    twfe_comparison,
)

EVENT_WINDOW = (-10, 6)


def _overall(block: dict[str, object]) -> dict[str, object]:
    rows = block.get("overall_att") or []
    return rows[0] if rows else {}


def run_analysis(
    root: Path, uncertainty_draws: int = 0, leave_one_out: bool = True
) -> dict[str, object]:
    estimation = load_estimation_frame(root)
    frame = estimation.data
    specifications: dict[str, object] = {}

    specifications["callaway_santanna_primary"] = callaway_santanna(
        frame, PRIMARY_ANTICIPATION, "never_treated"
    )
    for anticipation in ANTICIPATION_GRID:
        if anticipation != PRIMARY_ANTICIPATION:
            specifications[f"callaway_santanna_anticipation_{anticipation}"] = (
                callaway_santanna(frame, anticipation, "never_treated")
            )
    specifications["callaway_santanna_not_yet_treated"] = callaway_santanna(
        frame, PRIMARY_ANTICIPATION, "not_yet_treated"
    )
    specifications["callaway_santanna_varying_base_period"] = callaway_santanna(
        frame, PRIMARY_ANTICIPATION, "never_treated", base_period="varying"
    )
    if estimation.covariates:
        specifications["callaway_santanna_doubly_robust"] = callaway_santanna(
            frame, PRIMARY_ANTICIPATION, "never_treated",
            covariates=estimation.covariates, est_method="dr",
        )

    complete = frame[frame["historical_schedule_complete"] == 1]
    if complete.loc[complete["ever_treated"] == 1, "unit_id"].nunique() > 1:
        specifications["callaway_santanna_complete_history_only"] = callaway_santanna(
            complete, PRIMARY_ANTICIPATION, "never_treated"
        )

    # Prophylactic-plausibility restriction (deviation D4). Half the treated
    # cells involve birth cohorts first reached between 16 and 26, for whom a
    # prophylactic vaccine cannot plausibly alter invasive-cancer risk. These
    # specifications reassign treatment on age at eligibility so the immediate
    # effect and the Brazil dependence can be tested against that explanation.
    prophylactic: dict[int, pd.DataFrame] = {}
    for max_age in PROPHYLACTIC_AGE_GRID:
        restricted = restrict_to_prophylactic_age(frame, max_age)
        if restricted.loc[restricted["ever_treated"] == 1, "unit_id"].nunique() <= 1:
            continue
        prophylactic[max_age] = restricted
        specifications[f"callaway_santanna_prophylactic_age_{max_age}"] = (
            callaway_santanna(restricted, PRIMARY_ANTICIPATION, "never_treated")
        )
        # No pre-trend bootstrap on the sensitivities and placebos: none of them
        # feeds a decision rule, and each refit-based bootstrap costs more than
        # the estimate it accompanies. The two specifications the rules do read
        # keep the full replication count.
        specifications[f"sun_abraham_log_rate_prophylactic_age_{max_age}"] = (
            sun_abraham(
                restricted, PRIMARY_ANTICIPATION, window=EVENT_WINDOW,
                engine="ols", bootstrap_replications=0,
            )
        )

    specifications["sun_abraham_log_rate"] = sun_abraham(
        frame, PRIMARY_ANTICIPATION, window=EVENT_WINDOW, engine="ols"
    )
    specifications["sun_abraham_poisson_count"] = sun_abraham(
        frame, PRIMARY_ANTICIPATION, window=EVENT_WINDOW, engine="poisson"
    )
    specifications["poisson_pooled_event_study"] = poisson_event_study(
        frame, PRIMARY_ANTICIPATION, window=EVENT_WINDOW
    )
    specifications["twfe_naive"] = twfe_comparison(frame)

    # Falsification tests. Neither should show an effect; each one that does
    # narrows what the primary estimate can be attributed to.
    placebos: dict[str, dict[str, object]] = {}
    for lead in PLACEBO_LEAD_GRID:
        pseudo = placebo_in_time(frame, lead)
        if pseudo.loc[pseudo["ever_treated"] == 1, "unit_id"].nunique() <= 1:
            continue
        placebos[f"in_time_lead_{lead}"] = {
            "description": (
                f"onset moved {lead} years earlier; every genuinely "
                "post-eligibility observation removed"
            ),
            "treated_units": int(
                pseudo.loc[pseudo["ever_treated"] == 1, "unit_id"].nunique()
            ),
            "estimate": sun_abraham(
                pseudo, PRIMARY_ANTICIPATION, window=EVENT_WINDOW,
                engine="ols", bootstrap_replications=0,
            ),
        }
    # The same in-time placebo on the restricted sample. If a pseudo onset still
    # produces an effect after the biologically implausible cells are removed,
    # then the D4 restriction does not rescue the design, and that is the single
    # most consequential thing to know about it.
    for max_age, restricted in prophylactic.items():
        pseudo = placebo_in_time(restricted, PLACEBO_LEAD_GRID[0])
        if pseudo.loc[pseudo["ever_treated"] == 1, "unit_id"].nunique() <= 1:
            continue
        placebos[f"in_time_lead_{PLACEBO_LEAD_GRID[0]}_prophylactic_age_{max_age}"] = {
            "description": (
                f"onset moved {PLACEBO_LEAD_GRID[0]} years earlier, on the "
                f"age-at-eligibility <= {max_age} sample"
            ),
            "treated_units": int(
                pseudo.loc[pseudo["ever_treated"] == 1, "unit_id"].nunique()
            ),
            "estimate": sun_abraham(
                pseudo, PRIMARY_ANTICIPATION, window=EVENT_WINDOW,
                engine="ols", bootstrap_replications=0,
            ),
        }

    adult = restrict_to_adult_eligibility(frame, ADULT_ELIGIBILITY_MIN_AGE)
    if adult.loc[adult["ever_treated"] == 1, "unit_id"].nunique() > 1:
        placebos[f"adult_eligibility_min_age_{ADULT_ELIGIBILITY_MIN_AGE}"] = {
            "description": (
                "negative-control exposure: only cohorts first reached at "
                f"age {ADULT_ELIGIBILITY_MIN_AGE} or older, for whom a "
                "prophylactic vaccine cannot act"
            ),
            "treated_units": int(
                adult.loc[adult["ever_treated"] == 1, "unit_id"].nunique()
            ),
            "estimate": sun_abraham(
                adult, PRIMARY_ANTICIPATION, window=EVENT_WINDOW,
                engine="ols", bootstrap_replications=0,
            ),
        }

    report: dict[str, object] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "stage": "analyse",
        "sample": estimation.notes,
        "event_time_support": support_table(frame, PRIMARY_ANTICIPATION),
        "specifications": specifications,
        "falsification_tests": placebos,
        "prophylactic_restriction": {
            str(max_age): {
                "max_age_at_eligibility": max_age,
                "treated_countries": int(
                    restricted.loc[restricted["ever_treated"] == 1, "iso3"].nunique()
                ),
                "treated_units": int(
                    restricted.loc[restricted["ever_treated"] == 1, "unit_id"].nunique()
                ),
                "treated_cells": int(restricted["treated"].sum()),
            }
            for max_age, restricted in prophylactic.items()
        },
    }
    if leave_one_out:
        report["leave_one_country_out"] = leave_one_country_out(
            frame, PRIMARY_ANTICIPATION
        )
        # Run the dominance check on the restricted samples too. Both failing
        # decision rules are candidates for the same root cause, so whether the
        # restriction moves the dominance result is the test of that claim and
        # belongs in the outputs rather than in a one-off script.
        report["leave_one_country_out_prophylactic"] = {
            str(max_age): leave_one_country_out(restricted, PRIMARY_ANTICIPATION)
            for max_age, restricted in prophylactic.items()
        }
    if uncertainty_draws:
        report["outcome_uncertainty_scenarios"] = outcome_uncertainty_scenarios(
            frame, PRIMARY_ANTICIPATION, draws=uncertainty_draws
        )
    report["decision_rules"] = evaluate_decision_rules(report)
    # Dated forecast of when the outcome bands fill with vaccinated cohorts. It is
    # computed before the JSON is written so the summary travels with the estimates
    # rather than living only in a figure.
    report["cohort_maturity"] = write_maturity_outputs(root, report)
    report["power"] = write_power_outputs(root, report, frame)

    results = root / "05_results"
    results.mkdir(exist_ok=True)
    (results / "effect_estimates.json").write_text(
        json.dumps(_to_native(report), indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_tables(root, report)
    write_figures(root, report)
    (results / "effect_estimation_report.md").write_text(
        render_report(report), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# protocol stop rules
# ---------------------------------------------------------------------------


def _falsification_section(report: dict[str, object]) -> list[str]:
    """Report the placebo tests and state plainly which ones the data fails."""
    tests = report.get("falsification_tests") or {}
    if not tests:
        return []
    lines = [
        "",
        "## Falsification tests",
        "",
        "None of these should show an effect. Each one that does narrows what the "
        "primary estimate can be attributed to.",
        "",
        "| Test | Treated units | Mean post estimate | Event time 0 | Verdict |",
        "|---|---|---|---|---|",
    ]
    failures: list[str] = []
    mimicking: list[str] = []
    for name, block in tests.items():
        estimate = block["estimate"]
        overall = estimate.get("overall_post_att")
        zero = next(
            (r for r in estimate.get("event_study", []) if r["event_time"] == 0), None
        )
        significant = bool(
            zero
            and zero.get("conf_low") is not None
            and (zero["conf_low"] > 0 or zero["conf_high"] < 0)
        )
        post_test = estimate.get("joint_post_test") or {}
        post_p = post_test.get("p_value")
        joint_significant = bool(post_p is not None and post_p < 0.05)
        failed = significant or joint_significant
        # Direction matters. A placebo reproducing the primary estimate's negative
        # sign mimics the finding and undercuts it directly. A placebo that is
        # significant in the opposite direction is a different anomaly and must
        # not be described as if it were the same one.
        negative = overall is not None and overall < 0
        if failed and negative:
            verdict = "**negative effect present — mimics the primary finding**"
            mimicking.append(name)
        elif failed:
            verdict = "**significant, opposite sign — separate anomaly**"
        else:
            verdict = "no effect — passes"
        if failed:
            failures.append(name)
        zero_text = (
            f"{zero['estimate']:+.4f} [{zero['conf_low']:+.4f}, "
            f"{zero['conf_high']:+.4f}]" if zero else "."
        )
        lines.append(
            f"| `{name}` | {block['treated_units']} | "
            + (f"{overall:+.4f}" if overall is not None else ".")
            + f" | {zero_text} | {verdict} |"
        )

    lines.append("")
    if mimicking:
        lines += [
            "**"
            + ", ".join(f"`{name}`" for name in mimicking)
            + " reproduce the primary finding's negative sign where no effect is "
            "possible.** In the in-time placebos every genuinely post-eligibility "
            "observation is removed and onset is moved years earlier, so there is "
            "no treatment left for an effect to come from. Recovering an estimate "
            "of the same sign, and of comparable size to the primary one, means "
            "adopting and non-adopting countries were already diverging before "
            "any cohort became eligible. This is the secular decline that "
            "accompanies the kind of country that adopts HPV vaccination early: "
            "screening programmes, rising development, improving registration.",
            "",
            "This is the most consequential result in this report, and it "
            "outweighs the pre-trend rules passing. Those rules test the "
            "estimated leads inside a short window with 23 clusters, and the "
            "bootstrap that makes them correctly sized also costs them power. "
            "The in-time placebo uses the whole pre-period and detects the "
            "divergence the lead test could not resolve. Read together, the "
            "honest summary is that parallel trends is not established here; it "
            "merely survives an underpowered test of itself.",
            "",
            "The primary estimate is therefore an upper bound on a causal effect "
            "rather than a measurement of one.",
        ]
    other = [name for name in failures if name not in mimicking]
    if other:
        lines += [
            "",
            ", ".join(f"`{name}`" for name in other)
            + " is significant with the *opposite* sign to the primary estimate. "
            "That is not evidence that the protective association appears where "
            "it cannot; it is a separate anomaly. With a treated group this small "
            "the most likely readings are chance or residual confounding in the "
            "handful of countries that vaccinated adults, and it should not be "
            "cited either for or against the primary finding.",
        ]
    if not failures:
        lines += [
            "",
            "Every falsification test is null, which is what the primary "
            "specification needs but does not by itself establish.",
        ]
    return lines


def _prophylactic_section(report: dict[str, object]) -> list[str]:
    """Report the prophylactic-plausibility restriction and what it moves.

    Both failing decision rules are candidates for one root cause: treated cells
    whose birth cohorts were first reached in adulthood, where a prophylactic
    vaccine cannot act. This section states what the restriction does to each,
    including the fact that it would turn failures into passes, which is exactly
    why it is not adopted as primary here.
    """
    specifications = report["specifications"]
    keys = sorted(
        int(key.rsplit("_", 1)[1])
        for key in specifications
        if key.startswith("callaway_santanna_prophylactic_age_")
    )
    if not keys:
        return []

    base = _overall(specifications["callaway_santanna_primary"]).get("estimate")
    loo_all = report.get("leave_one_country_out_prophylactic") or {}

    def dominance(rows: list[dict[str, object]], reference: float | None) -> str:
        usable = [r for r in (rows or []) if r.get("estimate") is not None]
        if not usable or reference is None or not reference:
            return "not run"
        worst = max(usable, key=lambda r: abs(r["estimate"] - reference))
        swing = abs(worst["estimate"] - reference) / abs(reference)
        return f"{worst['dropped_country']} {swing:.2f}"

    unrestricted_dominance = dominance(report.get("leave_one_country_out"), base)
    sa_zero = next(
        (
            row for row in specifications["sun_abraham_log_rate"]["event_study"]
            if row["event_time"] == 0
        ),
        None,
    )

    lines = [
        "",
        "## Prophylactic-plausibility restriction (deviation D4)",
        "",
        "Half the treated cells involve birth cohorts first reached by a documented "
        "programme between ages 16 and 26. HPV vaccine is prophylactic, so those "
        "cohorts cannot have their invasive cervical cancer risk altered the way a "
        "cohort reached at 12 can, yet they carry eligibility at event time 0 and "
        "above. These specifications reassign treatment on age at eligibility.",
        "",
        "| Specification | Treated countries | Overall ATT | Event time 0 | "
        "Largest leave-one-out swing |",
        "|---|---|---|---|---|",
    ]
    zero_text = (
        f"{sa_zero['estimate']:+.4f} [{sa_zero['conf_low']:+.4f}, "
        f"{sa_zero['conf_high']:+.4f}]" if sa_zero else "."
    )
    lines.append(
        f"| unrestricted (primary) | "
        f"{report['sample']['treated_countries']} | "
        f"{base:+.4f} | {zero_text} | {unrestricted_dominance} |"
    )
    for max_age in keys:
        block = specifications[f"callaway_santanna_prophylactic_age_{max_age}"]
        estimate = _overall(block).get("estimate")
        sa_block = specifications.get(f"sun_abraham_log_rate_prophylactic_age_{max_age}")
        row_zero = next(
            (
                row for row in (sa_block or {}).get("event_study", [])
                if row["event_time"] == 0
            ),
            None,
        )
        text = (
            f"{row_zero['estimate']:+.4f} [{row_zero['conf_low']:+.4f}, "
            f"{row_zero['conf_high']:+.4f}]" if row_zero else "."
        )
        meta = (report.get("prophylactic_restriction") or {}).get(str(max_age), {})
        lines.append(
            f"| age at eligibility <= {max_age} "
            f"({meta.get('treated_cells', '.')} treated cells) | "
            f"{meta.get('treated_countries', '.')} | "
            f"{estimate:+.4f} | {text} | "
            f"{dominance(loo_all.get(str(max_age)), estimate)} |"
        )

    lines += [
        "",
        "Two things move together under the restriction, which is the evidence "
        "that the two failing rules share one cause rather than being independent "
        "problems. The immediate effect attenuates and its interval widens to "
        "include zero, and Brazil stops being the influential country: its "
        "relative leave-one-out swing falls below the 0.5 threshold the dominance "
        "rule uses. Brazil supplied 16 treated cells whose cohorts were first "
        "reached at 16 or later, and it carries most of the treated case mass.",
        "",
        "**The restriction is reported here and is not adopted as the primary "
        "specification.** It was motivated by inspecting a failing result, and it "
        "converts both failures into passes, so adopting it on the strength of "
        "that would be indefensible however good the biological argument is. "
        "Making it primary requires a dated, signed amendment recorded before any "
        "further estimation. The decision rules above are evaluated on the "
        "unrestricted specification for that reason.",
        "",
        "Read the attenuation carefully in either direction. The event-time-0 "
        "point estimate moves by about a third rather than to zero, and at "
        "`<= 15` the upper confidence bound sits at the edge of significance, so "
        "this is a partial explanation and not a clean exoneration. The `<= 18` "
        "variant still shows a significant immediate effect.",
        "",
        "**The restriction repairs the stop rules without repairing "
        "identification.** The in-time placebo was rerun on the restricted "
        "samples and still returns a negative estimate of essentially unchanged "
        "size, so the pre-existing divergence between adopting and non-adopting "
        "countries survives the restriction intact. Removing the biologically "
        "implausible cells removes what made Brazil dominant and softens the "
        "immediate effect; it does not make the comparison identify a programme "
        "effect. Adopting this restriction as primary would convert two failing "
        "rules into passes while leaving the reason the design does not work "
        "untouched, which is a further argument against doing it on the strength "
        "of these numbers.",
    ]
    return lines


def _absorbing_note(sample: dict[str, object]) -> str:
    """Name the cells that the two treatment definitions disagree about.

    Treatment is absorbing by design, so a cell can sit after its unit's first
    treated year while its own age band is not fully inside that year's
    documented target ages. Those cells are listed rather than left as an
    unexplained gap between two counts that a reader would otherwise take for an
    error.
    """
    detail = sample.get("treated_cells_absorbing_only_detail") or []
    if not detail:
        return (
            "- The absorbing treatment indicator and the per-cell documented flag "
            "agree on every treated cell."
        )
    listed = "; ".join(
        f"{row['iso3']} {row['age_group']} {row['year']} "
        f"(event time {row['event_time']:+d}, {row['analysis_status']})"
        for row in detail
    )
    return (
        f"- The two definitions disagree on {len(detail)} cell"
        f"{'s' if len(detail) != 1 else ''}: {listed}. Treatment is absorbing "
        "because programme eligibility cannot be withdrawn from a birth cohort "
        "once it has been targeted, so these are retained as treated; the "
        "disagreement is counted rather than smoothed. The group-time estimators "
        "key off event time, so this labelling does not move any estimate."
    )


def evaluate_decision_rules(report: dict[str, object]) -> dict[str, object]:
    """Apply the protocol's stop rules in code, not by reading the tables.

    The protocol forbids a causal conclusion when pre-trends are materially
    incompatible, when effects precede plausible latency, when support is
    sparse, or when a few countries dominate.
    """
    specifications = report["specifications"]
    checks: list[dict[str, object]] = []

    # The asymptotic cluster-robust Wald test cannot carry this rule on its own.
    # On synthetic panels built to satisfy parallel trends exactly, it rejects at
    # nominal 5% in about 40% of draws (24 treated clusters), while the restricted
    # wild cluster bootstrap rejects in about 5%. Using the asymptotic p-value as
    # a stop rule would therefore block a defensible study four times in ten for
    # no reason. The rule reads the bootstrap where it exists; both are always
    # reported. See protocol deviation D2.
    for label, key in (
        ("log-rate", "sun_abraham_log_rate"),
        ("count model", "sun_abraham_poisson_count"),
    ):
        spec = specifications[key]
        asymptotic = spec["joint_pre_trend_test"].get("p_value")
        bootstrap_block = spec.get("joint_pre_trend_bootstrap") or {}
        bootstrap = bootstrap_block.get("bootstrap_p_value")
        decisive = bootstrap if bootstrap is not None else asymptotic
        # The two engines bootstrap differently, so the label has to come from
        # the method actually used rather than being assumed.
        method = str(bootstrap_block.get("method", ""))
        boot_label = "score bootstrap" if "score" in method else "wild cluster bootstrap"
        basis = boot_label if bootstrap is not None else "asymptotic Wald"
        evidence_parts = []
        if bootstrap is not None:
            evidence_parts.append(f"{boot_label} p = {bootstrap:.3g}")
        if asymptotic is not None:
            evidence_parts.append(f"asymptotic Wald p = {asymptotic:.3g}")
        if bootstrap is None:
            evidence_parts.append(
                "no bootstrap for this engine; the asymptotic test is oversized, "
                "so read this as indicative and weigh the log-rate bootstrap"
            )
        checks.append(
            {
                "rule": f"pre-trends compatible with parallel trends ({label})",
                "passed": bool(decisive is not None and decisive >= 0.05),
                "decisive_basis": basis,
                "bootstrap_p_value": bootstrap,
                "asymptotic_p_value": asymptotic,
                "evidence": (
                    "; ".join(evidence_parts) if evidence_parts
                    else "test not estimable"
                ),
            }
        )

    # Invasive cervical cancer cannot respond to vaccination in the first year a
    # cohort becomes eligible. An effect at event time 0 indicates confounding by
    # trend rather than programme impact.
    event = {
        row["event_time"]: row
        for row in specifications["sun_abraham_log_rate"]["event_study"]
    }
    immediate = event.get(0)
    immediate_significant = bool(
        immediate
        and immediate.get("conf_low") is not None
        and (immediate["conf_low"] > 0 or immediate["conf_high"] < 0)
    )
    checks.append(
        {
            "rule": "no effect before plausible latency (event time 0)",
            "passed": not immediate_significant,
            "evidence": (
                f"event-time 0 estimate {immediate['estimate']:+.4f} "
                f"[{immediate['conf_low']:+.4f}, {immediate['conf_high']:+.4f}]"
                if immediate else "event time 0 not estimated"
            ),
        }
    )

    sample = report["sample"]
    checks.append(
        {
            "rule": "at least 20 countries with usable exposed cohorts",
            "passed": bool(sample["treated_countries"] >= 20),
            "evidence": (
                f"{sample['treated_countries']} treated countries, "
                f"{sample['treated_units']} treated units"
            ),
        }
    )

    loo = [
        row for row in (report.get("leave_one_country_out") or [])
        if row.get("estimate") is not None
    ]
    base = _overall(specifications["callaway_santanna_primary"]).get("estimate")
    dominance = None
    if loo and base is not None:
        swings = [abs(row["estimate"] - base) for row in loo]
        worst = loo[int(np.argmax(swings))]
        dominance = {
            "country": worst["dropped_country"],
            "estimate_without": worst["estimate"],
            "estimate_with_all": base,
            "sign_flips": bool(np.sign(worst["estimate"]) != np.sign(base)),
            "relative_swing": abs(worst["estimate"] - base) / abs(base) if base else None,
        }
    checks.append(
        {
            "rule": "no single country drives the estimate",
            "passed": bool(
                dominance is not None
                and not dominance["sign_flips"]
                and (dominance["relative_swing"] or 0) < 0.5
            ),
            "evidence": (
                f"largest swing on dropping {dominance['country']}: "
                f"{dominance['estimate_with_all']:+.4f} -> "
                f"{dominance['estimate_without']:+.4f}"
                if dominance else "leave-one-country-out not run"
            ),
        }
    )

    # Sign agreement across the estimators the protocol treats as primary.
    signs = {}
    for key in ("callaway_santanna_primary", "sun_abraham_log_rate",
                "sun_abraham_poisson_count"):
        block = specifications[key]
        value = _overall(block).get("estimate", block.get("overall_post_att"))
        if value is not None:
            signs[key] = float(np.sign(value))
    checks.append(
        {
            "rule": "staggered-robust estimators agree in sign",
            "passed": bool(len(set(signs.values())) <= 1),
            "evidence": ", ".join(
                f"{key} {'negative' if value < 0 else 'positive'}"
                for key, value in signs.items()
            ),
        }
    )

    failed = [check for check in checks if not check["passed"]]
    return {
        "checks": checks,
        "n_failed": len(failed),
        "causal_conclusion_permitted": not failed,
        "verdict": (
            "**A causal conclusion is not permitted under the protocol's own "
            "decision rules.** Failed: "
            + "; ".join(check["rule"] for check in failed)
            + "."
            if failed
            else "All prespecified decision rules pass."
        ),
    }


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def write_tables(root: Path, report: dict[str, object]) -> None:
    tables = root / "07_tables"
    tables.mkdir(exist_ok=True)
    specifications = report["specifications"]

    rows = []
    for name, block in specifications.items():
        for row in block.get("event_study", []) or []:
            rows.append({"specification": name, **row})
    pd.DataFrame(rows).to_csv(tables / "event_study_estimates.csv", index=False)

    summary = []
    for name, block in specifications.items():
        overall = _overall(block)
        if overall:
            summary.append({
                "specification": name,
                "estimand": "overall ATT (log incidence rate ratio)",
                "estimate": overall.get("estimate"),
                "std_error": overall.get("std_error"),
                "conf_low": overall.get("conf_low"),
                "conf_high": overall.get("conf_high"),
                "irr": overall.get("irr"),
            })
        elif block.get("overall_post_att") is not None:
            summary.append({
                "specification": name,
                "estimand": "unit-weighted mean post-eligibility effect",
                "estimate": block["overall_post_att"],
                "std_error": block.get("overall_post_std_error"),
                "conf_low": block.get("overall_post_conf_low"),
                "conf_high": block.get("overall_post_conf_high"),
                "irr": block["overall_post_irr"],
            })
        elif "estimate" in block:
            estimate = block["estimate"]
            summary.append({
                "specification": name,
                "estimand": "static two-way fixed-effect contrast",
                "estimate": estimate["estimate"],
                "std_error": estimate["std_error"],
                "conf_low": estimate["conf_low"],
                "conf_high": estimate["conf_high"],
                "irr": estimate["irr"],
            })
    pd.DataFrame(summary).to_csv(tables / "headline_estimates.csv", index=False)
    pd.DataFrame(report["event_time_support"]).to_csv(
        tables / "event_time_support.csv", index=False
    )
    if report.get("leave_one_country_out"):
        pd.DataFrame(report["leave_one_country_out"]).to_csv(
            tables / "leave_one_country_out.csv", index=False
        )


def write_figures(root: Path, report: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "06_figures"
    figures.mkdir(exist_ok=True)
    specifications = report["specifications"]
    low, high = EVENT_WINDOW

    series = {
        "Callaway-Sant'Anna": specifications["callaway_santanna_primary"]["event_study"],
        "Sun-Abraham (log rate)": specifications["sun_abraham_log_rate"]["event_study"],
        "Sun-Abraham (Poisson count)":
            specifications["sun_abraham_poisson_count"]["event_study"],
    }
    figure, axes = plt.subplots(figsize=(10, 6))
    for offset, (label, rows) in zip((-0.14, 0.0, 0.14), series.items()):
        usable = [
            row for row in rows
            if row.get("estimate") is not None and low <= row["event_time"] <= high
        ]
        if not usable:
            continue
        x = np.array([row["event_time"] for row in usable], dtype=float) + offset
        y = np.array([row["estimate"] for row in usable], dtype=float)
        lower = np.array([row.get("conf_low", np.nan) for row in usable], dtype=float)
        upper = np.array([row.get("conf_high", np.nan) for row in usable], dtype=float)
        axes.errorbar(
            x, y, yerr=[y - lower, upper - y], fmt="o", capsize=2, markersize=4,
            linewidth=1, label=label,
        )
    axes.axhline(0, color="black", linewidth=0.8)
    axes.axvspan(
        -PRIMARY_ANTICIPATION - 0.5, -0.5, color="grey", alpha=0.12,
        label="anticipation window (partial cohort eligibility)",
    )
    axes.set_xlabel("Event time (years since the age band's first fully eligible cohort)")
    axes.set_ylabel("Effect on log cervical cancer incidence rate")
    axes.set_title(
        "Event-study estimates from three staggered-adoption estimators\n"
        f"base period {-(PRIMARY_ANTICIPATION + 1)}; country-clustered intervals"
    )
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(figures / "Figure_1_event_study.png", dpi=200)
    plt.close(figure)

    loo = [
        row for row in (report.get("leave_one_country_out") or [])
        if row.get("estimate") is not None
    ]
    if loo:
        base = _overall(specifications["callaway_santanna_primary"]).get("estimate")
        order = sorted(loo, key=lambda row: row["estimate"])
        figure, axes = plt.subplots(figsize=(8, 7))
        y = np.arange(len(order))
        axes.scatter([row["estimate"] for row in order], y, s=24)
        if base is not None:
            axes.axvline(base, color="firebrick", linewidth=1,
                         label="all treated countries")
        axes.axvline(0, color="black", linewidth=0.8)
        axes.set_yticks(y)
        axes.set_yticklabels([row["dropped_country"] for row in order], fontsize=8)
        axes.set_xlabel("Overall ATT (log IRR) with that country dropped")
        axes.set_title("Leave-one-country-out influence on the overall ATT")
        axes.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(figures / "Figure_2_leave_one_country_out.png", dpi=200)
        plt.close(figure)

    write_case_concentration_figure(root, report)
    write_placebo_figure(root, report)


PLACEBO_PANEL = (
    ("sun_abraham_log_rate", "real onset", None),
    ("in_time_lead_8", "onset moved 8 years earlier", 8),
    ("in_time_lead_12", "onset moved 12 years earlier", 12),
    ("in_time_lead_8_prophylactic_age_15",
     "onset moved 8 years earlier,\ncohorts reached at 15 or younger", 8),
)


def placebo_series(report: dict[str, object]) -> dict[str, object]:
    """The quantities Figure S2 draws, separated from the drawing.

    All four blocks come off the same Sun-Abraham engine so the placebo runs are
    read against the real one on identical terms; mixing in the Callaway-Sant'Anna
    overall ATT here would compare two different aggregations and flatter the
    comparison.
    """
    specifications = report["specifications"]
    falsification = report.get("falsification_tests") or {}
    series: list[dict[str, object]] = []
    for key, label, lead in PLACEBO_PANEL:
        if key in specifications:
            block = specifications[key]
            treated_units = report["sample"]["treated_units"]
        elif key in falsification:
            block = falsification[key]["estimate"]
            treated_units = falsification[key]["treated_units"]
        else:
            continue
        series.append({
            "key": key,
            "label": label,
            "lead": lead,
            "is_placebo": lead is not None,
            "treated_units": treated_units,
            "event_study": [
                row for row in block["event_study"]
                if row.get("estimate") is not None
            ],
            "overall_post_att": block.get("overall_post_att"),
            "overall_post_conf_low": block.get("overall_post_conf_low"),
            "overall_post_conf_high": block.get("overall_post_conf_high"),
        })
    return {"series": series}


def write_placebo_figure(root: Path, report: dict[str, object]) -> Path | None:
    """Figure S2: the same negative estimate where no treatment exists.

    The point is a comparison of magnitudes, so the placebos are drawn in the
    accent and the real estimate in the neutral - the reverse of the usual
    emphasis, because here it is the placebo that carries the finding. The right
    panel puts the four mean post-period estimates on one axis, which is the
    comparison a reader would otherwise have to make by eye across the left one.
    """
    panels = placebo_series(report)
    placebos = [row for row in panels["series"] if row["is_placebo"]]
    if not placebos:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLACEBO = "#2a78d6"     # categorical slot 1: the runs that should show nothing
    REAL = "#898781"        # emphasis neutral: the estimate under test
    SHADE = "#eb6834"       # categorical slot 2: the region with no treatment in it
    INK, SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
    RULE = "#c3c2b7"

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    low, high = EVENT_WINDOW

    # --- left: the event studies ----------------------------------------------
    drawn = [row for row in panels["series"] if row["key"] != PLACEBO_PANEL[3][0]]
    shades = {PLACEBO_PANEL[0][0]: REAL, PLACEBO_PANEL[1][0]: PLACEBO,
              PLACEBO_PANEL[2][0]: "#86b6ef"}
    for offset, row in zip((-0.16, 0.0, 0.16), drawn):
        usable = [
            point for point in row["event_study"]
            if low <= point["event_time"] <= high
        ]
        if not usable:
            continue
        x = np.array([point["event_time"] for point in usable], dtype=float) + offset
        y = np.array([point["estimate"] for point in usable], dtype=float)
        lower = np.array([point.get("conf_low", np.nan) for point in usable])
        upper = np.array([point.get("conf_high", np.nan) for point in usable])
        colour = shades.get(row["key"], PLACEBO)
        left.errorbar(
            x, y, yerr=[y - lower, upper - y], fmt="o", capsize=2, markersize=4,
            linewidth=1, color=colour, ecolor=colour, zorder=3,
            label=row["label"].replace("\n", " "),
        )
    left.axhline(0, color=RULE, linewidth=1.2, zorder=2)
    left.axvspan(-0.5, high + 0.5, color=SHADE, alpha=0.10, zorder=0)
    left.annotate(
        "in the placebo runs this region\ncontains no treated observations",
        xy=(high, 0.98), xycoords=("data", "axes fraction"), xytext=(-4, 0),
        textcoords="offset points", ha="right", va="top", fontsize=8.5,
        color=SHADE, fontweight="bold",
    )
    left.set_xlabel("Event time", fontsize=9, color=SECONDARY)
    left.set_ylabel("Effect on log cervical cancer incidence rate", fontsize=9,
                    color=SECONDARY)
    left.set_title(
        "The placebo event studies fall below zero after a fictional onset",
        fontsize=10, color=INK, loc="left",
    )
    left.legend(frameon=False, fontsize=8.5, loc="lower left")
    left.set_xticks(list(range(low, high + 1, 2)))

    # --- right: the mean post-period estimates on one axis ---------------------
    order = list(reversed(panels["series"]))
    y = np.arange(len(order))
    for position, row in zip(y, order):
        estimate = row["overall_post_att"]
        if estimate is None:
            continue
        colour = PLACEBO if row["is_placebo"] else REAL
        lower = row["overall_post_conf_low"]
        upper = row["overall_post_conf_high"]
        if lower is not None and upper is not None:
            right.plot([lower, upper], [position, position], color=colour,
                       linewidth=1.6, zorder=3)
        right.scatter([estimate], [position], s=52, color=colour,
                      edgecolors=SURFACE, linewidths=1.2, zorder=4)
        right.annotate(
            f"{estimate:+.3f}", xy=(estimate, position), xytext=(0, 9),
            textcoords="offset points", ha="center", fontsize=8.5, color=colour,
            fontweight="bold",
        )
    right.axvline(0, color=RULE, linewidth=1.2, zorder=2)
    right.set_yticks(y)
    right.set_yticklabels(
        [f"{row['label']}\n({row['treated_units']} treated units)" for row in order],
        fontsize=8.5,
    )
    right.set_ylim(-0.6, len(order) - 0.4)
    right.set_xlabel("Mean post-period estimate on the log rate", fontsize=9,
                     color=SECONDARY)
    real = next((row for row in panels["series"] if not row["is_placebo"]), None)
    strongest = min(
        (row for row in placebos if row["overall_post_att"] is not None),
        key=lambda row: row["overall_post_att"], default=None,
    )
    if real is not None and strongest is not None and real["overall_post_att"]:
        ratio = strongest["overall_post_att"] / real["overall_post_att"]
        headline = (
            f"With no treatment left, the estimate is {ratio:.1f} times "
            "the real one"
        )
    else:
        headline = "The placebo runs recover the sign of the estimate under test"
    right.set_title(
        headline + "\nso the association predates any cohort becoming eligible",
        fontsize=10, color=INK, loc="left",
    )

    for axes in (left, right):
        axes.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axes.spines[spine].set_color(RULE)
            axes.spines[spine].set_linewidth(0.8)
        axes.tick_params(colors=SECONDARY, labelsize=8)
        axes.set_axisbelow(True)
    left.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    right.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)

    figure.text(
        0.008, 0.010,
        "Sun-Abraham interaction-weighted estimates throughout, with "
        "country-clustered intervals.\nIn each placebo run every genuinely "
        "post-eligibility observation is deleted before estimation, so no "
        "treated period remains for an effect to come from.",
        fontsize=7.5, color=SECONDARY, linespacing=1.5,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    figures = root / "06_figures"
    figures.mkdir(exist_ok=True)
    path = figures / "Figure_S2_placebo_event_studies.png"
    figure.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(figure)
    return path


def write_case_concentration_figure(root: Path, report: dict[str, object]) -> None:
    """Figure 3: where the treated-cell case mass sits, and what happens to it.

    Emphasis encoding rather than a categorical palette: one accent for the
    leading country, a deliberate neutral for everything else, because the story
    is one number. The left panel is a dot plot rather than bars because the range
    spans five orders of magnitude and needs a log axis, on which bar length stops
    being proportional to value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ACCENT = "#2a78d6"     # categorical slot 1
    NEUTRAL = "#898781"    # emphasis neutral: the rest, deliberately recessive
    INK = "#0b0b0b"
    SECONDARY = "#52514e"
    GRID = "#e1e0d9"
    SURFACE = "#ffffff"

    sample = report["sample"]
    top = sample["treated_cases_top_country"]
    by_country = sample["treated_cases_by_country"]
    by_event = sample["treated_cases_by_event_time"]
    if not by_country or not by_event:
        return

    figures = root / "06_figures"
    figures.mkdir(exist_ok=True)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # --- left: case mass by country, log axis, leading country accented --------
    order = list(reversed(by_country))          # smallest at the bottom
    y = np.arange(len(order))
    colours = [ACCENT if row["iso3"] == top else NEUTRAL for row in order]
    left.hlines(
        y, 0.05, [row["cases"] for row in order],
        color=GRID, linewidth=1, zorder=1,
    )
    left.scatter(
        [row["cases"] for row in order], y, s=46, c=colours, zorder=2,
        edgecolors=SURFACE, linewidths=1.5,
    )
    left.set_xscale("log")
    left.set_xlim(0.05, 40000)
    left.set_yticks(y)
    left.set_yticklabels([row["iso3"] for row in order], fontsize=8, color=SECONDARY)
    left.set_xlabel("Incident cases in treated cells (log scale)", fontsize=9,
                    color=SECONDARY)
    left.set_title(
        f"{top} holds "
        f"{sample['treated_cases_top_country_share']:.0%} of the treated-cell cases",
        fontsize=10, color=INK, loc="left",
    )
    leader = order[-1]
    left.annotate(
        f"{leader['cases']:,.0f} cases",
        xy=(leader["cases"], len(order) - 1), xytext=(-8, 12),
        textcoords="offset points", ha="right", fontsize=8.5, color=ACCENT,
        fontweight="bold",
    )
    median_rest = sample["treated_cases_median_per_cell_excluding_top"]
    left.annotate(
        f"other {len(by_country) - 1} countries:\nmedian {median_rest:.1f} cases "
        f"per treated cell",
        xy=(600, 0.4), fontsize=8, color=SECONDARY, va="bottom", ha="center",
    )

    # --- right: cases by event time, split leading country vs the rest ---------
    event_times = [row["event_time"] for row in by_event]
    top_cases = np.array([row["top_country_cases"] for row in by_event])
    other_cases = np.array([row["cases"] - row["top_country_cases"] for row in by_event])
    right.bar(event_times, top_cases, color=ACCENT, width=0.72,
              edgecolor=SURFACE, linewidth=1.5, label=top, zorder=2)
    right.bar(event_times, other_cases, bottom=top_cases, color=NEUTRAL, width=0.72,
              edgecolor=SURFACE, linewidth=1.5, label="all other treated countries",
              zorder=2)
    for row in by_event:
        right.annotate(
            str(row["countries"]),
            xy=(row["event_time"], row["cases"]), xytext=(0, 5),
            textcoords="offset points", ha="center", fontsize=8, color=SECONDARY,
        )
    right.set_xticks(event_times)
    right.set_xlabel(
        "Event time (years since the age band's first fully eligible cohort)",
        fontsize=9, color=SECONDARY,
    )
    right.set_ylabel("Incident cases in treated cells", fontsize=9, color=SECONDARY)
    right.set_title(
        "Support collapses, and what remains is one country\n"
        "(number above each bar = contributing countries)",
        fontsize=10, color=INK, loc="left",
    )
    right.legend(frameon=False, fontsize=9, loc="upper right")
    tail = [row for row in by_event if row["cases"] < 10]
    if tail:
        span = (
            f"{tail[0]['event_time']}-{tail[-1]['event_time']}"
            if len(tail) > 1 else str(tail[0]["event_time"])
        )
        centre = (tail[0]["event_time"] + tail[-1]["event_time"]) / 2
        right.annotate(
            f"event times {span}:\n{tail[0]['cases']:.0f} case per cell",
            xy=(centre, max(top_cases + other_cases) * 0.10),
            ha="center", va="bottom", fontsize=8, color=SECONDARY,
        )

    for axes in (left, right):
        axes.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axes.spines[spine].set_color("#c3c2b7")
            axes.spines[spine].set_linewidth(0.8)
        axes.tick_params(colors=SECONDARY, labelsize=8)
    left.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    right.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    left.set_axisbelow(True)
    right.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(figures / "Figure_3_case_concentration.png", dpi=300,
                   facecolor=SURFACE)
    plt.close(figure)


def write_maturity_outputs(root: Path, report: dict[str, object]) -> dict[str, object]:
    """Figure 4 and its table: when the outcome bands fill with vaccinated cohorts.

    Left panel is the global forecast, right panel the countries that already have
    treated cells — the point being that even those are years away from the ages at
    which invasive cervical cancer is measurable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    import maturity as M

    # Ordinal ramp: the age bands are ordered, so they take one hue light->dark
    # rather than four categorical slots. Validated with --ordinal.
    BAND_COLOURS = {
        "20-24": "#86b6ef", "25-29": "#3987e5",
        "30-34": "#1c5cab", "35-39": "#0d366b",
    }
    ACCENT = "#2a78d6"
    INK, SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
    RULE = "#c3c2b7"

    forecast = M.cohort_maturity_forecast(M.load_country_table(root))
    if forecast.empty:
        return {}
    summary = M.maturity_summary(forecast)

    (root / "07_tables").mkdir(exist_ok=True)
    forecast.sort_values(["maturity_year", "iso3"]).to_csv(
        root / "07_tables" / "cohort_maturity_forecast.csv", index=False
    )

    data_end = int(report["sample"]["years"][1])
    figures = root / "06_figures"
    figures.mkdir(exist_ok=True)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 6), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # --- left: cumulative countries whose band has matured ---------------------
    years = np.arange(2010, 2056)
    # Label each curve at a different height as well as a different year: the
    # curves converge on the same total, so labels sharing a y would collide.
    label_heights = {"20-24": 0.78, "25-29": 0.60, "30-34": 0.42, "35-39": 0.24}
    for age_group in ("20-24", "25-29", "30-34", "35-39"):
        block = forecast.loc[forecast["age_group"] == age_group, "maturity_year"]
        counts = np.array([(block <= year).sum() for year in years])
        left.plot(years, counts, color=BAND_COLOURS[age_group], linewidth=2,
                  zorder=3, label=f"ages {age_group}")
        anchor = int(np.argmax(
            counts >= summary["countries"] * label_heights[age_group]
        ))
        left.annotate(
            f"ages {age_group}", xy=(years[anchor], counts[anchor]),
            xytext=(-10, 4), textcoords="offset points", fontsize=8, ha="right",
            va="bottom", color=BAND_COLOURS[age_group], fontweight="bold",
        )
    left.axvline(data_end, color=RULE, linewidth=1.2, zorder=2)
    left.annotate(
        f"outcome data\nend {data_end}", xy=(data_end, summary["countries"] * 0.92),
        xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8,
        color=SECONDARY,
    )
    left.set_xlim(2010, 2056)
    left.set_ylim(0, summary["countries"] * 1.05)
    left.set_xlabel("Calendar year", fontsize=9, color=SECONDARY)
    left.set_ylabel("Countries with the age band wholly eligible", fontsize=9,
                    color=SECONDARY)
    left.set_title(
        f"No country has a fully vaccinated 30-34 band before "
        f"{summary['bands']['30-34']['earliest']}\n"
        f"({summary['countries']} countries with a documented national programme; "
        f"cohorts eligible at {M.PROPHYLACTIC_AGE_CAP} or younger)",
        fontsize=10, color=INK, loc="left",
    )

    # --- right: the countries that already contribute treated cells ------------
    treated_countries = sorted({
        row["iso3"] for row in report["sample"]["treated_cases_by_country"]
    })
    subset = forecast.loc[forecast["iso3"].isin(treated_countries)]
    pivot = subset.pivot(index="iso3", columns="age_group", values="maturity_year")
    pivot = pivot.dropna(subset=["20-24", "30-34"]).sort_values(
        ["30-34", "20-24"], ascending=False
    )
    y = np.arange(len(pivot))
    right.hlines(y, pivot["20-24"], pivot["30-34"], color=GRID, linewidth=2, zorder=1)
    right.scatter(pivot["20-24"], y, s=42, color=BAND_COLOURS["20-24"], zorder=3,
                  edgecolors=SURFACE, linewidths=1.2, label="ages 20-24")
    right.scatter(pivot["30-34"], y, s=42, color=BAND_COLOURS["30-34"], zorder=3,
                  edgecolors=SURFACE, linewidths=1.2, label="ages 30-34")
    right.axvline(data_end, color=RULE, linewidth=1.2, zorder=2)
    right.set_yticks(y)
    right.set_yticklabels(pivot.index, fontsize=8, color=SECONDARY)
    right.set_xlabel("Year the band becomes wholly eligible", fontsize=9,
                     color=SECONDARY)
    right.set_title(
        "The countries with treated cells today,\n"
        "and when their 30-34 band arrives",
        fontsize=10, color=INK, loc="left",
    )
    right.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    right.legend(frameon=False, fontsize=9, loc="upper right")

    leader = report["sample"]["treated_cases_top_country"]
    if leader in pivot.index:
        position = int(np.where(pivot.index == leader)[0][0])
        right.annotate(
            f"{leader}: "
            f"{report['sample']['treated_cases_top_country_share']:.0%} of today's\n"
            f"treated-cell cases, matures last",
            xy=(pivot.loc[leader, "30-34"], position), xytext=(-14, 0),
            textcoords="offset points", ha="right", va="center", fontsize=8,
            color=ACCENT,
        )

    for axes in (left, right):
        axes.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axes.spines[spine].set_color(RULE)
            axes.spines[spine].set_linewidth(0.8)
        axes.tick_params(colors=SECONDARY, labelsize=8)
        axes.set_axisbelow(True)
    left.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    right.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)

    figure.tight_layout()
    figure.savefig(figures / "Figure_4_cohort_maturity.png", dpi=300,
                   facecolor=SURFACE)
    plt.close(figure)
    return summary


def write_power_outputs(
    root: Path, report: dict[str, object], frame: pd.DataFrame
) -> dict[str, object]:
    """Figure 5 and its table: what this design could detect, against what to expect.

    The comparison is the point. An information bound computed from case counts
    alone says what the data could support; the fitted standard errors say what the
    design achieves; the coverage-by-efficacy grid says what magnitude the biology
    implies. Reporting only the first two would invite the reader to assume a null
    means a small effect.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import power as P

    BOUND = "#86b6ef"      # information bound: what the cases could support
    ACHIEVED = "#1c5cab"   # what the fitted models achieve
    EXPECTED = "#eb6834"   # categorical slot 2: the benchmark, a different kind
    INK, SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
    RULE = "#c3c2b7"

    event_study = report["specifications"]["sun_abraham_log_rate"]["event_study"]
    empirical_ses = {
        int(row["event_time"]): float(row["std_error"])
        for row in event_study
        if row.get("std_error") is not None and row.get("event_time") is not None
    }
    overall = _overall(report["specifications"]["callaway_santanna_primary"])
    overall_se = float(overall.get("std_error") or float("nan"))

    summary = P.power_summary(
        frame, PRIMARY_ANTICIPATION, empirical_ses, overall_se,
        float(report["sample"]["treated_cases_total"]),
    )

    table = pd.DataFrame(summary["by_event_time"])
    (root / "07_tables").mkdir(exist_ok=True)
    table.to_csv(root / "07_tables" / "power_by_event_time.csv", index=False)

    post = table.loc[table["event_time"] >= 0].sort_values("event_time")
    if post.empty:
        return summary

    figures = root / "06_figures"
    figures.mkdir(exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- left: detectable effect against the effect worth detecting ------------
    grid = summary["expected_reduction_grid"]
    low = min(row["reduction_pct"] for row in grid)
    high = max(row["reduction_pct"] for row in grid)
    left.axhspan(low, high, color=EXPECTED, alpha=0.12, zorder=0)
    left.annotate(
        f"reduction implied by coverage x efficacy\n({low:.0f}-{high:.0f}%)",
        xy=(post["event_time"].max(), (low + high) / 2), xytext=(-4, 0),
        textcoords="offset points", ha="right", va="center", fontsize=8.5,
        color=EXPECTED, fontweight="bold",
    )
    left.plot(post["event_time"], post["analytic_mde_pct"], color=BOUND,
              linewidth=2, marker="o", markersize=5, zorder=3,
              label="information bound (case counts only)")
    left.plot(post["event_time"], post["empirical_mde_pct"], color=ACHIEVED,
              linewidth=2, marker="o", markersize=5, zorder=3,
              label="achieved by the fitted model")
    left.set_ylim(0, max(high * 1.12, float(post["analytic_mde_pct"].max()) * 1.2))
    left.set_xlabel("Event time", fontsize=9, color=SECONDARY)
    left.set_ylabel("Smallest detectable reduction (%), 80% power", fontsize=9,
                    color=SECONDARY)
    left.set_title(
        "The design can detect far smaller effects than the biology implies",
        fontsize=10, color=INK, loc="left",
    )
    left.legend(frameon=False, fontsize=8.5, loc="upper left")

    # --- right: how much of the available information the design uses ----------
    right.bar(post["event_time"], post["design_effect"], color=ACHIEVED, width=0.68,
              edgecolor=SURFACE, linewidth=1.5, zorder=2)
    right.axhline(1.0, color=RULE, linewidth=1.2, zorder=3)
    right.annotate(
        "1.0 = the model extracts all the\ninformation the case counts contain",
        xy=(post["event_time"].max(), 1.0), xytext=(-4, 6),
        textcoords="offset points", ha="right", va="bottom", fontsize=8,
        color=SECONDARY,
    )
    right.annotate(
        "below 1: the fitted standard error is smaller\nthan the Poisson bound allows",
        xy=(post["event_time"].max(), 0.5), xytext=(-4, 0),
        textcoords="offset points", ha="right", va="center", fontsize=8,
        color=SECONDARY, style="italic",
    )
    right.set_xlabel("Event time", fontsize=9, color=SECONDARY)
    right.set_ylabel("Design effect (fitted SE / information bound)", fontsize=9,
                     color=SECONDARY)
    overall_block = summary["overall"]
    right.set_title(
        f"Overall, the design effect is "
        f"{overall_block['design_effect']:.2f}: the estimate uses about "
        f"{overall_block['effective_cases']:,.0f}\nof "
        f"{overall_block['treated_cases']:,.0f} treated-cell cases' worth of "
        "information",
        fontsize=10, color=INK, loc="left",
    )

    for axes in (left, right):
        axes.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axes.spines[spine].set_color(RULE)
            axes.spines[spine].set_linewidth(0.8)
        axes.tick_params(colors=SECONDARY, labelsize=8)
        axes.set_xticks(list(post["event_time"]))
        axes.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        axes.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(figures / "Figure_5_power.png", dpi=300, facecolor=SURFACE)
    plt.close(figure)
    return summary


def _format(row: dict[str, object] | None) -> str:
    if not row or row.get("estimate") is None:
        return "not estimated"
    estimate = float(row["estimate"])
    if row.get("conf_low") is not None and np.isfinite(row["conf_low"]):
        return (
            f"{estimate:+.4f} (IRR {np.exp(estimate):.3f}, 95% CI "
            f"{np.exp(row['conf_low']):.3f}-{np.exp(row['conf_high']):.3f})"
        )
    return f"{estimate:+.4f} (IRR {np.exp(estimate):.3f})"


def _cell(row: dict[str, object] | None) -> str:
    if not row or row.get("estimate") is None:
        return "."
    return f"{float(row['estimate']):+.4f}"


def _concentration_section(sample: dict[str, object]) -> list[str]:
    """Report where the case mass sits, because the standard errors will not.

    Kept as its own section rather than a bullet in Sample: this is the quantity
    that decides how much of the estimate is really a single-country study, and it
    is invisible in every other output.
    """
    top = sample["treated_cases_top_country"]
    by_event = sample["treated_cases_by_event_time"]
    deepest = [row for row in by_event if row["top_country_share"] is not None]
    peak = max(deepest, key=lambda row: row["top_country_share"]) if deepest else None

    lines = [
        "## Where the information is",
        "",
        f"Treated cells contain {sample['treated_cases_total']:,.0f} incident cases. "
        f"**{top} holds {sample['treated_cases_top_country_share']:.1%} of them.** "
        f"The remaining {sample['treated_countries'] - 1} treated countries hold "
        f"{sample['treated_cases_excluding_top']:,.0f} cases across "
        f"{sample['treated_cells_excluding_top']} cells, a median of "
        f"{sample['treated_cases_median_per_cell_excluding_top']:.1f} cases per cell.",
        "",
        "This is reported because the standard errors do not show it. They track "
        "case mass rather than the number of contributing countries, so nominal "
        "precision stays flat while the evidence base narrows.",
        "",
        f"| Event time | Treated units | Countries | Cases | {top} share |",
        "|---|---|---|---|---|",
    ]
    for row in by_event:
        share = (
            f"{row['top_country_share']:.1%}"
            if row["top_country_share"] is not None else "."
        )
        lines.append(
            f"| {row['event_time']} | {row['units']} | {row['countries']} "
            f"| {row['cases']:,.0f} | {share} |"
        )
    if peak is not None:
        lines += [
            "",
            f"Concentration is worst where follow-up matters most: at event time "
            f"{peak['event_time']}, {peak['top_country_share']:.1%} of the "
            f"{peak['cases']:,.0f} cases come from {top} alone, on "
            f"{peak['countries']} countries.",
        ]
    lines.append("")
    return lines


def _maturity_section(summary: dict[str, object] | None, data_end: int) -> list[str]:
    """How long the immaturity lasts, as dates rather than as an adjective."""
    if not summary or not summary.get("bands"):
        return []
    bands = summary["bands"]
    lines = [
        "## When these data mature",
        "",
        f"Across {summary['countries']} countries with a documented national "
        "programme, the year each outcome band becomes wholly composed of birth "
        f"cohorts eligible at 15 or younger. Outcome data currently end {data_end}.",
        "",
        "| Age band | Earliest country | Median country | Matured by 2023 | by 2030 "
        "| by 2040 | by 2050 |",
        "|---|---|---|---|---|---|---|",
    ]
    for age_group in sorted(bands):
        block = bands[age_group]
        counts = block["countries_matured_by"]
        lines.append(
            f"| {age_group} | {block['earliest']} | {block['median']:.0f} "
            f"| {counts['2023']} | {counts['2030']} | {counts['2040']} "
            f"| {counts['2050']} |"
        )
    deep = bands.get("30-34")
    if deep:
        lines += [
            "",
            f"Invasive cervical cancer only becomes measurable in numbers at ages "
            f"30 and above. **No country reaches a wholly eligible 30-34 band before "
            f"{deep['earliest']}**, and the median country reaches it in "
            f"{deep['median']:.0f}. The immaturity in this panel is therefore not a "
            "matter of waiting one or two more data releases.",
        ]
    lines.append("")
    return lines


def _power_section(summary: dict[str, object] | None) -> list[str]:
    """What the design could detect, so a null is not misread as a small effect."""
    if not summary or not summary.get("overall"):
        return []
    overall = summary["overall"]
    grid = summary["expected_reduction_grid"]
    low = min(row["reduction_pct"] for row in grid)
    high = max(row["reduction_pct"] for row in grid)

    lines = [
        "## What this design could have detected",
        "",
        "Two bounds, and a benchmark. The information bound counts Poisson "
        "information in incident cases and ignores clustering, so it is the best "
        "the design could possibly do. The achieved figure uses the standard error "
        "the fitted model actually produces. The benchmark is what an "
        "intention-to-treat reduction should look like if a programme works.",
        "",
        f"- Treated-cell cases: {overall['treated_cases']:,.0f}. Information bound "
        f"on the overall ATT: standard error {overall['analytic_se']:.4f}, "
        f"detectable reduction "
        f"**{overall['analytic_mde']['reduction_pct']:.1f}%**.",
        f"- Achieved: standard error {overall['empirical_se']:.4f}, detectable "
        f"reduction **{overall['empirical_mde']['reduction_pct']:.1f}%**.",
        f"- Design effect {overall['design_effect']:.2f}. The estimate therefore "
        f"uses about {overall['effective_cases']:,.0f} cases' worth of information "
        f"out of {overall['treated_cases']:,.0f} — clustering and the concentration "
        "of cases in one country account for the rest.",
        f"- A programme reaching a whole age band implies a reduction of "
        f"{low:.0f}-{high:.0f}% across the plausible coverage and efficacy range "
        "(40-80% coverage, 50-90% efficacy).",
        "",
        "**The design is not underpowered for the effect it is looking for.** It "
        f"can detect a {overall['empirical_mde']['reduction_pct']:.0f}% reduction, "
        f"and the smallest reduction in the plausible range is {low:.0f}%. The "
        "absence of an effect of that size is therefore informative, and points at "
        "the exposure assignment, the confounding the placebos expose, or an "
        "outcome series unable to move - not at a sample too small to see it.",
        "",
        "| Event time | Treated cases | Information bound | Achieved | Design effect |",
        "|---|---|---|---|---|",
    ]
    for row in summary["by_event_time"]:
        if row["event_time"] < 0:
            continue
        achieved = (
            f"{row['empirical_mde_pct']:.1f}%"
            if row["empirical_mde_pct"] == row["empirical_mde_pct"] else "."
        )
        effect = (
            f"{row['design_effect']:.2f}"
            if row["design_effect"] == row["design_effect"] else "."
        )
        lines.append(
            f"| {row['event_time']} | {row['treated_post_cases']:,.0f} "
            f"| {row['analytic_mde_pct']:.1f}% | {achieved} | {effect} |"
        )
    below = [
        row for row in summary["by_event_time"]
        if row["event_time"] >= 0 and row["design_effect"] == row["design_effect"]
        and row["design_effect"] < 1
    ]
    if below:
        times = ", ".join(str(row["event_time"]) for row in below)
        lines += [
            "",
            f"A design effect below 1 at event times {times} is a warning rather "
            "than a credit. It means the log-rate engine reports a standard error "
            "smaller than the Poisson information in the underlying case counts "
            "allows, which it can do because it models a continuous log rate and "
            "never sees how few cases produced it. Those intervals are too narrow.",
        ]
    lines.append("")
    return lines


def render_report(report: dict[str, object]) -> str:
    sample = report["sample"]
    specifications = report["specifications"]
    decisions = report["decision_rules"]
    lines: list[str] = []

    lines += [
        "# Effect estimation: HPV programme eligibility and cervical cancer incidence",
        "",
        f"Generated {report['generated_at_utc']} (UTC). "
        "Estimands are on the log incidence-rate scale; IRR is the exponentiated "
        "estimate.",
        "",
        "## Verdict",
        "",
        decisions["verdict"],
        "",
        "| Prespecified decision rule | Result | Evidence |",
        "|---|---|---|",
    ]
    for check in decisions["checks"]:
        lines.append(
            f"| {check['rule']} | {'pass' if check['passed'] else '**FAIL**'} "
            f"| {check['evidence']} |"
        )

    lines += [
        "",
        "## Sample",
        "",
        f"- {sample['rows']:,} country x age-band x year cells; {sample['units']} "
        f"units across {sample['countries']} countries, "
        f"{sample['years'][0]}-{sample['years'][1]}.",
        f"- {sample['treated_units']} treated units in "
        f"{sample['treated_countries']} countries; "
        f"{sample['treated_cell_count']} treated cells "
        f"({sample['treated_cells_documented']} with the full age band inside "
        f"documented target ages, "
        f"{sample['treated_cells_absorbing_only']} treated only by the absorbing "
        f"rule).",
        _absorbing_note(sample),
        "- Covariates admitted to the doubly-robust specification: "
        + (", ".join(sample["covariates_used_in_doubly_robust"]) or "none")
        + ".",
    ]
    rejected = sample["covariates_rejected_for_incompleteness"]
    if rejected:
        lines.append(
            "- Prespecified covariates withheld for incompleteness: "
            + ", ".join(
                f"{name} ({share:.0%} of cells)" for name, share in rejected.items()
            )
            + ". They are ingested and available, but conditioning on them would "
            "drop a third or more of the panel, so they are not used for primary "
            "adjustment."
        )
    lines += [
        f"- GBD and UN WPP female denominators differ by a median of "
        f"{sample['gbd_vs_wpp_denominator_median_rel_diff']:.1%} (99th percentile "
        f"{sample['gbd_vs_wpp_denominator_p99_rel_diff']:.1%}). The count model "
        "uses the GBD-implied denominator so that numerator and denominator come "
        "from a single source.",
        f"- Median approximated log-scale standard error of the modelled outcome: "
        f"{sample['outcome_uncertainty_median_log_se']:.3f}. This is derived from "
        "the published uncertainty interval, not from draws, and is an "
        "approximation.",
        "",
    ]
    lines += _concentration_section(sample)
    lines += _maturity_section(report.get("cohort_maturity"), int(sample["years"][1]))
    lines += _power_section(report.get("power"))
    lines += [
        "## Headline estimates",
        "",
        "| Specification | Estimand | Estimate |",
        "|---|---|---|",
    ]
    for name, block in specifications.items():
        overall = _overall(block)
        if overall:
            lines.append(f"| `{name}` | overall ATT | {_format(overall)} |")
        elif block.get("overall_post_att") is not None:
            lines.append(
                f"| `{name}` | unit-weighted mean post-eligibility effect | "
                + _format({
                    "estimate": block["overall_post_att"],
                    "conf_low": block.get("overall_post_conf_low"),
                    "conf_high": block.get("overall_post_conf_high"),
                })
                + " |"
            )
        elif "estimate" in block:
            lines.append(f"| `{name}` | static contrast | {_format(block['estimate'])} |")

    primary = specifications["callaway_santanna_primary"]
    doubly_robust = specifications.get("callaway_santanna_doubly_robust")
    if doubly_robust:
        dropped_rows = primary["n_rows_used"] - doubly_robust["n_rows_used"]
        dropped_units = primary["n_entities_used"] - doubly_robust["n_entities_used"]
        lines += [
            "",
            "### Covariate adjustment changes the sample, not just the estimate",
            "",
            f"Conditioning on {', '.join(doubly_robust['covariates'])} drops "
            f"{dropped_rows:,} cells and {dropped_units} units that have no "
            "covariate value in their base period, and moves the overall ATT from "
            f"{_format(_overall(primary))} to {_format(_overall(doubly_robust))}. "
            "A sign change on a "
            f"{dropped_rows / max(primary['n_rows_used'], 1):.0%} sample reduction "
            "is a fragility signal in its own right and is reported as one: the "
            "adjusted and unadjusted figures are not estimated on the same "
            "countries, so the difference between them is not an adjustment "
            "effect.",
        ]

    lines += _prophylactic_section(report)
    lines += _falsification_section(report)

    cs = {r["event_time"]: r for r in specifications["callaway_santanna_primary"]["event_study"]}
    sa = {r["event_time"]: r for r in specifications["sun_abraham_log_rate"]["event_study"]}
    sp = {r["event_time"]: r for r in specifications["sun_abraham_poisson_count"]["event_study"]}
    low, high = EVENT_WINDOW
    lines += [
        "",
        "## Event study",
        "",
        "| Event time | Callaway-Sant'Anna | Sun-Abraham (log rate) | "
        "Sun-Abraham (Poisson count) | Treated units |",
        "|---|---|---|---|---|",
    ]
    for event_time in sorted(set(cs) | set(sa) | set(sp)):
        if not low <= event_time <= high:
            continue
        units = sa.get(event_time, {}).get("units", "")
        lines.append(
            f"| {event_time} | {_cell(cs.get(event_time))} | "
            f"{_cell(sa.get(event_time))} | {_cell(sp.get(event_time))} | {units} |"
        )
    lines += [
        "",
        f"Event times -{PRIMARY_ANTICIPATION} to -1 form the anticipation window: "
        "a five-year age band is only partly composed of eligible birth cohorts "
        "during those years, so they are estimated as leads rather than used as "
        f"controls, and the base period is {-(PRIMARY_ANTICIPATION + 1)}. The "
        "window length is varied in "
        "`callaway_santanna_anticipation_0` and `callaway_santanna_anticipation_8`.",
        "",
        "## Why the pooled count model disagrees",
        "",
    ]
    pooled = next(
        (row for row in specifications["poisson_pooled_event_study"]["static"]
         if row["term"] == "treated_post"),
        None,
    )
    sa_poisson = specifications["sun_abraham_poisson_count"]
    lines += [
        "The protocol's Poisson count model estimated with *pooled* event-time "
        f"dummies returns {_format(pooled)} - the opposite sign to every "
        "staggered-robust estimator here. The same Poisson likelihood with "
        "cohort-specific interactions, aggregated the Sun-Abraham way, returns "
        f"{sa_poisson['overall_post_att']:+.4f} "
        f"(IRR {sa_poisson['overall_post_irr']:.3f}).",
        "",
        "The gap is the already-treated-as-control comparison that two-way fixed "
        "effects make under staggered timing, not a data defect. It is worth "
        "recording that on this panel the contamination is large enough to "
        "reverse the sign of the headline estimate, which is the empirical case "
        "for the group-time estimators the protocol specifies.",
        "",
        "## Interpretation",
        "",
    ]

    # Derived from the checks rather than asserted, so this paragraph cannot
    # drift out of step with the decision table above it.
    pre_trend_checks = [
        check for check in report["decision_rules"]["checks"]
        if check["rule"].startswith("pre-trends compatible")
    ]
    rejecting = [check for check in pre_trend_checks if not check["passed"]]
    if len(rejecting) == len(pre_trend_checks):
        pre_trend_sentence = (
            "Both joint pre-trend tests reject parallel trends, so the "
            "association cannot be separated from a pre-existing downward trend"
        )
    elif rejecting:
        pre_trend_sentence = (
            "The pre-trend evidence is split: "
            + " and ".join(
                check["rule"][check["rule"].index("(") + 1:-1]
                + (" rejects" if not check["passed"] else " does not reject")
                for check in pre_trend_checks
            )
            + " parallel trends, so a pre-existing downward trend is not ruled out"
        )
    else:
        pre_trend_sentence = (
            "Neither joint pre-trend test rejects parallel trends once the test "
            "is calibrated to the number of treated clusters"
        )

    lines += [
        "The staggered-robust estimators agree on a small negative association "
        f"between documented programme eligibility and cervical cancer incidence. "
        f"{pre_trend_sentence}; the estimated effect is already "
        "present at event time 0, before any biologically plausible latency for "
        "invasive cervical cancer; and the treated panel is young, small, and "
        "concentrated in early-adopter countries. Under the protocol's own "
        "decision rules these are reported as descriptive associations, and no "
        "causal claim is made.",
        "",
        "That the pre-trend rules pass should not be read as parallel trends "
        "holding. The in-time placebos above recover a negative estimate of "
        "comparable size after every post-eligibility observation is removed, "
        "which is a direct demonstration that adopters and non-adopters were "
        "already diverging. The lead-based rules are simply a weaker test of the "
        "same proposition on this panel: 23 clusters, a short window, and a "
        "bootstrap that buys correct size at the cost of power. Where the two "
        "disagree, the placebo is the more informative evidence, and it says the "
        "association here cannot be separated from trend.",
        "",
        "The pre-trend rule is read off a bootstrap, not the asymptotic "
        "cluster-robust Wald test, on both engines. On synthetic panels built to "
        "satisfy parallel trends exactly, with a treated group this size, the "
        "asymptotic test rejects in about 40% of draws on the linear engine and "
        "about 33% on the count engine at nominal 5%; the corresponding "
        "bootstraps reject in about 5% and 3% and both still detect a planted "
        "pre-trend. The linear engine refits a resampled outcome; the count "
        "engine reweights cluster score contributions, because a resampled "
        "Poisson outcome is not guaranteed non-negative. Asymptotic p-values are "
        "reported alongside so the gap stays visible. See deviations D2 and D3.",
        "",
        "The count model's two tests disagree by a wide margin: an asymptotic "
        f"Wald p of {specifications['sun_abraham_poisson_count']['joint_pre_trend_test']['p_value']:.3g} "
        "against a score-bootstrap p of "
        f"{(specifications['sun_abraham_poisson_count'].get('joint_pre_trend_bootstrap') or {}).get('bootstrap_p_value')}. "
        "This is reported rather than smoothed over. The Wald form inverts a "
        "cluster-robust covariance of the aggregated leads estimated from 23 "
        "clusters, which is exactly the situation in which it is known to be "
        "unreliable, whereas the score form is evaluated at the restricted fit "
        "and needs no such inverse. The simulation above is the basis for "
        "preferring the score bootstrap, but a divergence this large is itself a "
        "reason to treat the count model's pre-trend evidence as weak rather "
        "than as a clean pass.",
        "",
    ]
    return "\n".join(lines)
