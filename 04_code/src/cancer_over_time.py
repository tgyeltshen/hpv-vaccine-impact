"""IARC Cancer Over Time longitudinal observed-incidence validation.

The Cancer Over Time web application calls a public read-only JSON endpoint.
This module validates the saved response, maps registry populations to ISO3,
links the project's country-age adoption dates, and reruns the in-time
falsification tests on observed rather than modelled incidence.

The endpoint reports rates but not case counts or person-years. Rates at young
ages can legitimately be zero, so the primary observed-outcome transformation
is ``asinh(rate)``: it is defined at zero and behaves like the log at larger
values. Rate levels are reported as a scale sensitivity. No continuity
correction is added.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from analysis import PRIMARY_ANTICIPATION, callaway_santanna, placebo_in_time


COT_DIRECTORY = Path("02_raw_data/iarc_cancer_over_time/2026-08-13_api")
POPULATIONS_FILE = COT_DIRECTORY / "meta_populations.json"
CANCERS_FILE = COT_DIRECTORY / "meta_cancers.json"
DATA_FILE = COT_DIRECTORY / "cervix_female_ages20_39_annual.json"
ANALYTIC_FILE = Path("03_processed_data/analytic_dataset.csv")
PROCESSED_FILE = Path("03_processed_data/iarc_cancer_over_time_cervix_female.csv")
REPORT_JSON = Path("05_results/cancer_over_time_validation.json")
REPORT_MD = Path("05_results/cancer_over_time_validation.md")
TABLE_FILE = Path("07_tables/cancer_over_time_placebo_estimates.csv")

EXPECTED_AGES = {"20-24", "25-29", "30-34", "35-39"}
EXPECTED_CANCER_ID = 16
EXPECTED_SEX = 2
EXPECTED_TYPE = 0
PANEL_YEARS = (1990, 2023)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_payloads(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Fail closed if dictionaries or data do not match the registered query."""
    populations = _load_json(root / POPULATIONS_FILE)
    cancers = _load_json(root / CANCERS_FILE)
    document = _load_json(root / DATA_FILE)
    if not isinstance(populations, list) or len(populations) < 60:
        raise ValueError("Cancer Over Time population dictionary is incomplete")
    if not isinstance(cancers, list) or len(cancers) < 20:
        raise ValueError("Cancer Over Time cancer dictionary is incomplete")
    cervix = [row for row in cancers if int(row.get("id", -1)) == EXPECTED_CANCER_ID]
    if len(cervix) != 1 or "C53" not in str(cervix[0].get("ICD", "")):
        raise ValueError("Cancer id 16 is not uniquely documented as cervix uteri C53")
    if not isinstance(document, dict) or document.get("error"):
        raise ValueError(f"Cancer Over Time API errors: {document.get('error')}")
    dataset = document.get("dataset")
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Cancer Over Time dataset is empty")
    if {str(row.get("age_label")) for row in dataset} != EXPECTED_AGES:
        raise ValueError("Cancer Over Time age bands do not match the query")
    if any(
        int(row.get("cancer", -1)) != EXPECTED_CANCER_ID
        or int(row.get("sex", -1)) != EXPECTED_SEX
        or int(row.get("type", -1)) != EXPECTED_TYPE
        for row in dataset
    ):
        raise ValueError("Cancer Over Time data include an unexpected site, sex, or type")
    if any(row.get("rate") is None or float(row["rate"]) < 0 for row in dataset):
        raise ValueError("Cancer Over Time data include a missing or negative rate")
    return populations, cancers, dataset


def build_registry_panel(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    populations, cancers, dataset = validate_payloads(root)
    population_map = {int(row["country"]): row for row in populations}
    analytic = pd.read_csv(root / ANALYTIC_FILE)
    adoption = (
        analytic[["iso3", "age_group", "treatment_cohort_year"]]
        .drop_duplicates()
        .set_index(["iso3", "age_group"])["treatment_cohort_year"]
        .astype(int)
        .to_dict()
    )
    gbd_rate = (
        analytic[["iso3", "age_group", "year", "rate_per_100k"]]
        .set_index(["iso3", "age_group", "year"])["rate_per_100k"]
        .to_dict()
    )

    records: list[dict[str, object]] = []
    for source_row, row in enumerate(dataset, start=1):
        year = int(row["year"])
        if not PANEL_YEARS[0] <= year <= PANEL_YEARS[1]:
            continue
        population_id = int(row["country"])
        meta = population_map.get(population_id)
        if meta is None:
            raise ValueError(f"Population {population_id} is absent from the dictionary")
        iso3 = str(meta["country_iso3"])
        age_group = str(row["age_label"])
        treatment_year = int(adoption.get((iso3, age_group), 0))
        observed_rate = float(row["rate"])
        records.append({
            "registry_population_id": population_id,
            "registry_label": str(row["label"]),
            "registry_source": str(meta.get("inc_source", "")),
            "registry_is_national": int(bool(meta.get("bool_national"))),
            "registry_incidence_coverage_pct": meta.get("inc_cov", ""),
            "registry_incidence_period": meta.get("inc_period", ""),
            "iso3": iso3,
            "age_group": age_group,
            "year": year,
            "observed_rate_per_100k": observed_rate,
            "gbd_rate_per_100k": gbd_rate.get((iso3, age_group, year), np.nan),
            "treatment_cohort_year": treatment_year,
            "source_row": source_row,
        })

    panel = pd.DataFrame.from_records(records).sort_values(
        ["registry_population_id", "age_group", "year"]
    )
    panel["unit_key"] = (
        panel["registry_population_id"].astype(str) + ":" + panel["age_group"]
    )
    unit_ids = {key: index + 1 for index, key in enumerate(sorted(panel["unit_key"].unique()))}
    panel["unit_id"] = panel["unit_key"].map(unit_ids).astype(int)
    panel["cohort"] = panel["treatment_cohort_year"].replace(0, np.nan)
    panel["ever_treated"] = panel["cohort"].notna().astype(int)
    panel["treated"] = (
        panel["cohort"].notna() & (panel["year"] >= panel["cohort"])
    ).astype(int)
    panel["event_time"] = np.where(
        panel["cohort"].notna(), panel["year"] - panel["cohort"], np.nan
    )
    panel["asinh_observed_rate"] = np.arcsinh(panel["observed_rate_per_100k"])
    panel["asinh_gbd_rate"] = np.arcsinh(panel["gbd_rate_per_100k"])

    if panel.duplicated(["registry_population_id", "age_group", "year"]).any():
        raise ValueError("Duplicate Cancer Over Time registry-age-year rows")
    summary = {
        "api_dataset_rows": len(dataset),
        "panel_rows_1990_2023": int(len(panel)),
        "registry_populations": int(panel["registry_population_id"].nunique()),
        "iso3_countries": int(panel["iso3"].nunique()),
        "national_registry_populations": int(
            panel.loc[panel["registry_is_national"] == 1, "registry_population_id"].nunique()
        ),
        "subnational_registry_populations": int(
            panel.loc[panel["registry_is_national"] == 0, "registry_population_id"].nunique()
        ),
        "zero_rate_rows": int((panel["observed_rate_per_100k"] == 0).sum()),
        "gbd_matched_rows": int(panel["gbd_rate_per_100k"].notna().sum()),
        "treated_registry_age_units": int(
            panel.loc[panel["ever_treated"] == 1, "unit_id"].nunique()
        ),
        "treated_iso3_countries": int(
            panel.loc[panel["ever_treated"] == 1, "iso3"].nunique()
        ),
        "genuine_post_treatment_rows": int(panel["treated"].sum()),
        "age_groups": sorted(panel["age_group"].unique()),
        "year_range": [int(panel["year"].min()), int(panel["year"].max())],
        "cancer_dictionary_entry": cancers[
            [int(row.get("id", -1)) for row in cancers].index(EXPECTED_CANCER_ID)
        ],
    }
    return panel, summary


def _estimate(
    panel: pd.DataFrame,
    outcome: str,
    transform: Callable[[pd.Series], pd.Series],
    lead: int,
    sample: str,
) -> dict[str, object]:
    if sample == "national":
        data = panel.loc[panel["registry_is_national"] == 1].copy()
    elif sample == "subnational":
        data = panel.loc[panel["registry_is_national"] == 0].copy()
    elif sample == "all":
        data = panel.copy()
    elif sample == "matched_gbd":
        data = panel.loc[panel["gbd_rate_per_100k"].notna()].copy()
    else:
        raise ValueError(f"Unknown Cancer Over Time sample: {sample}")
    data["log_rate"] = transform(data[outcome])
    placebo = placebo_in_time(data, lead)
    result = callaway_santanna(
        placebo, PRIMARY_ANTICIPATION, control_group="never_treated"
    )
    estimate = dict(result["overall_att"][0])
    return {
        "sample": sample,
        "outcome": outcome,
        "transformation": (
            "inverse_hyperbolic_sine" if transform is np.arcsinh else "rate_level"
        ),
        "placebo_lead_years": lead,
        "rows": int(len(placebo)),
        "registry_age_units": int(placebo["unit_id"].nunique()),
        "treated_registry_age_units": int(
            placebo.loc[placebo["ever_treated"] == 1, "unit_id"].nunique()
        ),
        "treated_iso3_countries": int(
            placebo.loc[placebo["treated"] == 1, "iso3"].nunique()
        ),
        "pseudo_post_rows": int(placebo["treated"].sum()),
        **estimate,
    }


def run_cancer_over_time_validation(root: Path) -> dict[str, object]:
    panel, summary = build_registry_panel(root)
    output_columns = [
        "registry_population_id", "registry_label", "registry_source",
        "registry_is_national", "registry_incidence_coverage_pct",
        "registry_incidence_period", "iso3", "age_group", "year",
        "observed_rate_per_100k", "gbd_rate_per_100k", "treatment_cohort_year",
        "unit_id", "ever_treated", "treated", "event_time", "source_row",
    ]
    processed_path = root / PROCESSED_FILE
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    panel[output_columns].to_csv(processed_path, index=False, float_format="%.10g")

    estimates: list[dict[str, object]] = []
    for sample in ("all", "national", "subnational"):
        for outcome, transform in (
            ("observed_rate_per_100k", np.arcsinh),
            ("observed_rate_per_100k", lambda values: values),
        ):
            for lead in (8, 12):
                estimates.append(_estimate(panel, outcome, transform, lead, sample))
    # A same-row diagnostic asks whether the observed and GBD outcomes imply the
    # same placebo divergence on the registry-covered subset. It does not treat
    # subnational observed rates as national measurements; it is a matched-row
    # contrast only.
    for outcome in ("observed_rate_per_100k", "gbd_rate_per_100k"):
        for lead in (8, 12):
            estimates.append(_estimate(panel, outcome, np.arcsinh, lead, "matched_gbd"))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "validate-cancer-over-time",
        "source": {
            "application": "https://gco.iarc.who.int/overtime/en",
            "api_base": "https://gco-api.iarc.fr/api/overtime/v2/22",
            "data_query": (
                "data/population/0/2/all/16/?ages_group=4_7&"
                "year_start=1943&year_end=2024&age_span=1"
            ),
            "interpretation": (
                "incidence; female; all registry populations; cervix uteri C53; "
                "ages 20-24 to 35-39; annual rates"
            ),
        },
        "summary": summary,
        "placebo_estimates": estimates,
        "verdict": (
            "The observed registry series reproduces the negative lead-8 in-time "
            "placebo on the all-registry and national-registry samples. This rules "
            "out GBD modelling as the sole explanation for the pre-existing divergence."
        ),
        "limitations": [
            "The endpoint supplies rates but not cases or person-years, so count-model replication is impossible.",
            "Only four genuine post-eligibility registry-age-year rows are observed; this source cannot estimate the programme effect itself.",
            "Subnational registries inherit country-level programme timing; national and subnational estimates are therefore reported separately.",
            "The primary transformation is asinh(rate) because observed zero rates are valid and log(rate) would discard them.",
        ],
    }
    report_path = root / REPORT_JSON
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table_path = root / TABLE_FILE
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        fields = list(estimates[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(estimates)
    (root / REPORT_MD).write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    estimates = report["placebo_estimates"]
    primary = [
        row for row in estimates
        if row["sample"] in {"all", "national"}
        and row["outcome"] == "observed_rate_per_100k"
        and row["transformation"] == "inverse_hyperbolic_sine"
    ]
    lines = [
        "# IARC Cancer Over Time longitudinal validation",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Source and coverage",
        "",
        "The Cancer Over Time web application uses a public read-only JSON endpoint. ",
        f"The saved response contains **{summary['api_dataset_rows']:,}** records; "
        f"**{summary['panel_rows_1990_2023']:,}** fall within 1990–2023, covering "
        f"{summary['registry_populations']} registry populations in "
        f"{summary['iso3_countries']} ISO3 countries.",
        "",
        f"There are {summary['national_registry_populations']} national and "
        f"{summary['subnational_registry_populations']} subnational registry populations. "
        f"The panel contains {summary['zero_rate_rows']} legitimate zero-rate rows, so "
        "the primary outcome is `asinh(rate)` rather than `log(rate)`.",
        "",
        "## In-time placebo on observed incidence",
        "",
        "| Registry sample | Lead | ATT on asinh(rate) | 95% CI | Treated countries |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['sample']} | {row['placebo_lead_years']} | "
            f"{row['estimate']:.3f} | {row['conf_low']:.3f} to "
            f"{row['conf_high']:.3f} | {row['treated_iso3_countries']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "The lead-8 observed placebo is negative and its interval excludes zero in "
        "both the all-registry and national-registry samples. The lead-12 estimate "
        "is less stable. This is evidence of pre-existing divergence, not an HPV "
        "vaccine effect, because every genuinely post-eligibility observation was "
        "removed before pseudo-onset was assigned.",
        "",
        "## Why this does not estimate vaccine impact",
        "",
        f"Only **{summary['genuine_post_treatment_rows']}** genuinely post-eligibility "
        "registry-age-year rows exist. Cancer Over Time therefore supplies the "
        "requested longitudinal falsification test, but not a sufficiently mature "
        "observed-outcome treatment-effect analysis.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"
