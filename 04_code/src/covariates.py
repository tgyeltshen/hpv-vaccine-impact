"""Ingest covariate sources into a country-year panel.

Every field keeps its source status (``modelled``, ``survey_estimated``,
``country_reported``, ``derived``) per rule 6 of the package README. Nothing is
interpolated: a missing country-year stays missing so that the analysis code
decides how to handle it explicitly.

Three properties of the WHO/GHO payloads are handled explicitly here, because
each one silently destroys a field if it is ignored:

1. Sex dimension. Cervical indicators are reported on ``Dim1='SEX_FMLE'``
   rather than on the both-sexes code, so a both-sexes-only filter drops the
   entire series.
2. Value type. ``NCD_CCS_cervicalcancerpgmcvg`` is published as ordered
   *bands* ("10 to 50"), not as a number; ``NumericValue`` is null on every
   row. It is read as a label, and the band midpoint is exposed separately as
   an explicitly ``derived`` field.
3. Non-response codes. "No data received", "No response" and "Don't know" are
   survey non-response, not observations, and are mapped to missing rather
   than ingested as category levels.

Each field also carries a ``time_varying`` flag in the returned statistics.
Several GHO series are one or a few survey rounds; a field observed in a
single year cannot identify anything in a model that already includes unit
fixed effects, and the analysis stage reads this flag rather than rediscovering
the problem.
"""

from __future__ import annotations

import json
from pathlib import Path


# World Bank ``country/all`` returns regional and income aggregates alongside
# countries. They are removed by keeping only ISO3 codes in the analytic
# crosswalk rather than by pattern-matching aggregate names.
WORLDBANK_INDICATORS = {
    "NY.GDP.PCAP.PP.KD": ("gdp_per_capita_ppp", "derived"),
    "SP.URB.TOTL.IN.ZS": ("urban_population_pct", "derived"),
    "SE.SEC.ENRR": ("secondary_enrolment_gross_pct", "derived"),
    "SH.DYN.AIDS.ZS": ("hiv_prevalence_15_49_pct", "modelled"),
    "SH.PRV.SMOK.FE": ("female_tobacco_use_pct", "modelled"),
}

# ``sex_codes`` lists the ``Dim1`` values that carry the wanted series. ``None``
# means the indicator is not sex-disaggregated and only undimensioned rows are
# accepted.
GHO_INDICATORS = {
    "UHC_INDEX_REPORTED": ("uhc_service_coverage_index", "modelled", None),
    "NCD_CXCA_SCREENED_WITHIN_TIMEPERIOD": (
        "cervical_screening_prevalence_30_49_pct",
        "survey_estimated",
        ("SEX_FMLE", "FMLE"),
    ),
}

# Indicators published as labels. ``bands`` maps a label to a numeric value for
# the companion ``*_pct``/``_flag`` field; labels absent from the map are kept
# as text but contribute no number.
GHO_CATEGORICAL = {
    "NCD_CCS_cervicalcancerscreening": (
        "national_cervical_screening_programme",
        "country_reported",
        None,
        {"Yes": 1.0, "No": 0.0},
        "national_cervical_screening_programme_flag",
    ),
    "NCD_CCS_cervicalcancerpgmcvg": (
        "cervical_screening_programme_coverage_band",
        "country_reported",
        None,
        {
            "Less than 10": 5.0,
            "10 to 50": 30.0,
            "more than 50 but less than 70": 60.0,
            "70 or more": 80.0,
        },
        "cervical_screening_programme_coverage_midpoint_pct",
    ),
}

# Survey administration outcomes, not measurements. "Not applicable" is a real
# programme fact (no programme to report coverage for) so the label is kept,
# but it yields no number: coding it as zero coverage would assert something
# the respondent did not report.
GHO_NON_RESPONSE = frozenset(
    {"No data received", "No response", "Don't know", "Not reported", ""}
)


def read_worldbank(path: Path, keep_iso3: set[str]) -> dict[tuple[str, int], float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or len(document) < 2 or not document[1]:
        raise ValueError(f"World Bank payload has no observations: {path.name}")
    values: dict[tuple[str, int], float] = {}
    for row in document[1]:
        iso3 = (row.get("countryiso3code") or "").strip()
        if iso3 not in keep_iso3 or row.get("value") is None:
            continue
        values[(iso3, int(row["date"]))] = float(row["value"])
    return values


def _accepts_sex(row: dict, sex_codes: tuple[str, ...] | None) -> bool:
    dimension = row.get("Dim1")
    if sex_codes is None:
        return dimension in (None, "", "BTSX", "SEX_BTSX")
    return dimension in sex_codes


def read_gho_numeric(
    path: Path, keep_iso3: set[str], sex_codes: tuple[str, ...] | None = None
) -> dict[tuple[str, int], float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values: dict[tuple[str, int], float] = {}
    for row in document.get("value", []):
        iso3 = (row.get("SpatialDim") or "").strip()
        if iso3 not in keep_iso3 or row.get("TimeDim") is None:
            continue
        if (row.get("SpatialDimType") or "COUNTRY") != "COUNTRY":
            continue
        if not _accepts_sex(row, sex_codes):
            continue
        numeric = row.get("NumericValue")
        if numeric is None:
            continue
        values[(iso3, int(row["TimeDim"]))] = float(numeric)
    return values


def read_gho_categorical(
    path: Path, keep_iso3: set[str], sex_codes: tuple[str, ...] | None = None
) -> dict[tuple[str, int], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values: dict[tuple[str, int], str] = {}
    for row in document.get("value", []):
        iso3 = (row.get("SpatialDim") or "").strip()
        if iso3 not in keep_iso3 or row.get("TimeDim") is None:
            continue
        if (row.get("SpatialDimType") or "COUNTRY") != "COUNTRY":
            continue
        if not _accepts_sex(row, sex_codes):
            continue
        label = (row.get("Value") or "").strip()
        if not label or label in GHO_NON_RESPONSE:
            continue
        values[(iso3, int(row["TimeDim"]))] = label
    return values


def build_covariate_panel(
    worldbank_dir: Path, gho_dir: Path, keep_iso3: set[str]
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[str, object]]:
    panel: dict[tuple[str, int], dict[str, object]] = {}
    stats: dict[str, object] = {"fields": {}}

    def assign(field: str, values: dict[tuple[str, int], object], status: str) -> None:
        for key, value in values.items():
            panel.setdefault(key, {})[field] = value
        years = sorted({key[1] for key in values})
        stats["fields"][field] = {
            "observations": len(values),
            "countries": len({key[0] for key in values}),
            "years_observed": len(years),
            "year_range": [years[0], years[-1]] if years else [],
            "data_status": status,
            # A single observed year cannot be separated from a unit fixed
            # effect. The analysis stage refuses such fields as adjustment
            # covariates instead of letting them be absorbed silently.
            "time_varying": len(years) > 1,
        }

    for indicator, (field, status) in WORLDBANK_INDICATORS.items():
        assign(field, read_worldbank(worldbank_dir / f"{indicator}.json", keep_iso3), status)

    for indicator, (field, status, sex_codes) in GHO_INDICATORS.items():
        assign(
            field,
            read_gho_numeric(gho_dir / f"{indicator}.json", keep_iso3, sex_codes),
            status,
        )

    for indicator, spec in GHO_CATEGORICAL.items():
        field, status, sex_codes, bands, derived_field = spec
        labels = read_gho_categorical(gho_dir / f"{indicator}.json", keep_iso3, sex_codes)
        assign(field, labels, status)
        numeric = {key: bands[value] for key, value in labels.items() if value in bands}
        assign(derived_field, numeric, "derived")
        stats["fields"][derived_field]["derived_from"] = field
        stats["fields"][derived_field]["band_midpoints"] = bands
        stats["fields"][field]["labels_without_number"] = sorted(
            {value for value in labels.values() if value not in bands}
        )

    stats["country_years"] = len(panel)
    stats["countries"] = len({key[0] for key in panel})
    stats["non_response_codes_dropped"] = sorted(GHO_NON_RESPONSE - {""})
    return panel, stats


COVARIATE_FIELDS = (
    [field for field, _ in WORLDBANK_INDICATORS.values()]
    + [field for field, _, _ in GHO_INDICATORS.values()]
    + [name for spec in GHO_CATEGORICAL.values() for name in (spec[0], spec[4])]
)
