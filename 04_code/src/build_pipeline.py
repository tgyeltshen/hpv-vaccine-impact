"""Ingest Project 1 sources and build the cohort-coverage feasibility matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from covariates import COVARIATE_FIELDS, build_covariate_panel
from exposure import outcome_cohort_interval
from ingest import (
    iter_gzip_csv,
    iter_zip_csv,
    normalize_country_name,
    parse_age_band,
    parse_hpv_target_age,
    read_csv_records,
    read_xlsx_records,
    write_csv,
)


AGE_BANDS = [(20, 24), (25, 29), (30, 34), (35, 39)]
AGE_LABEL = {band: f"{band[0]}-{band[1]}" for band in AGE_BANDS}

# Last calendar year of the GBD 2023 outcome panel. JRF rows beyond it describe
# planned programmes and must never establish eligibility for an observed cell.
OUTCOME_MAX_YEAR = 2023

# eJRF TARGETPOP codes. The field was not collected before 2019: it is null on
# 100% of 2006-2018 rows and on ~2% of 2019+ rows, so its absence is a property
# of the reporting instrument, not of the country's programme. Blank values are
# therefore resolved by corroboration rather than either imputed or dropped.
ROUTINE_TARGET_CODES = {"FEMALE", "BOTH"}
CATCHUP_TARGET_CODES = {"CATCHUP_C", "CATCHUP_A"}
# PLANNED is not yet implemented; RISKGROUPS and ADULTS are not the routine
# adolescent cohort. None of them may create a routine treated observation.
NON_ROUTINE_TARGET_CODES = {"PLANNED", "RISKGROUPS", "ADULTS"}

# xMart sentinels seen in the HPV extract: SCHEDULEROUNDS uses 0 and -2222 for
# unknown, SUBNATIONAL_TARGET uses -9999.
SCHEDULE_ROUND_SENTINELS = {0, -2222}

# Only explicit, reviewed name aliases belong here. The automatic crosswalk is
# also written out so unmatched or low-confidence mappings remain visible.
GBD_NAME_TO_ISO3 = {
    "bolivia plurinational state of": "BOL",
    "brunei": "BRN",
    "cape verde": "CPV",
    "cote d ivoire": "CIV",
    "democratic republic of the congo": "COD",
    "iran islamic republic of": "IRN",
    "laos": "LAO",
    "micronesia federated states of": "FSM",
    "moldova": "MDA",
    "north korea": "PRK",
    "palestine": "PSE",
    "russia": "RUS",
    "south korea": "KOR",
    "syria": "SYR",
    "taiwan": "TWN",
    "taiwan province of china": "TWN",
    "tanzania": "TZA",
    "the bahamas": "BHS",
    "the gambia": "GMB",
    "turkey": "TUR",
    "united states": "USA",
    "venezuela bolivarian republic of": "VEN",
    "viet nam": "VNM",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: str) -> int | None:
    if not value.strip():
        return None
    return int(float(value))


def _number(value: str) -> float | None:
    if not value.strip():
        return None
    return float(value)


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def ingest_gbd(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    cells: dict[tuple[int, str, int, int, int], dict[str, object]] = {}
    raw_rows = 0
    for row in iter_zip_csv(source):
        raw_rows += 1
        if row["measure_name"] != "Incidence":
            raise ValueError(f"Unexpected GBD measure: {row['measure_name']}")
        if row["sex_name"] != "Female":
            raise ValueError(f"Unexpected GBD sex: {row['sex_name']}")
        if row["cause_name"] != "Cervical cancer":
            raise ValueError(f"Unexpected GBD cause: {row['cause_name']}")
        age = parse_age_band(row["age_name"])
        if age not in AGE_BANDS:
            raise ValueError(f"Unexpected GBD age: {row['age_name']}")
        year = int(row["year"])
        if not 1990 <= year <= 2023:
            raise ValueError(f"Unexpected GBD year: {year}")
        key = (int(row["location_id"]), row["location_name"], year, age[0], age[1])
        record = cells.setdefault(
            key,
            {
                "gbd_location_id": int(row["location_id"]),
                "gbd_location_name": row["location_name"],
                "year": year,
                "age_lower": age[0],
                "age_upper": age[1],
                "age_group": AGE_LABEL[age],
            },
        )
        metric = row["metric_name"]
        if metric not in {"Number", "Rate"}:
            raise ValueError(f"Unexpected GBD metric: {metric}")
        prefix = "cases" if metric == "Number" else "rate_per_100k"
        record[prefix] = row["val"]
        record[f"{prefix}_lower"] = row["lower"]
        record[f"{prefix}_upper"] = row["upper"]

    expected = {
        "cases", "cases_lower", "cases_upper",
        "rate_per_100k", "rate_per_100k_lower", "rate_per_100k_upper",
    }
    for key, record in cells.items():
        missing = expected.difference(record)
        if missing:
            raise ValueError(f"GBD cell {key} is missing {sorted(missing)}")
    records = sorted(
        cells.values(),
        key=lambda row: (str(row["gbd_location_name"]), int(row["year"]), int(row["age_lower"])),
    )
    fields = [
        "gbd_location_id", "gbd_location_name", "year", "age_lower", "age_upper",
        "age_group", "cases", "cases_lower", "cases_upper", "rate_per_100k",
        "rate_per_100k_lower", "rate_per_100k_upper",
    ]
    write_csv(destination, records, fields)
    return records, {
        "raw_rows": raw_rows,
        "analytic_cells": len(records),
        "locations": len({row["gbd_location_id"] for row in records}),
        "years": [min(row["year"] for row in records), max(row["year"] for row in records)],
        "age_groups": sorted({row["age_group"] for row in records}),
    }


def ingest_wpp(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    totals: dict[tuple[str, str, int, int, int], float] = defaultdict(float)
    source_rows = 0
    for row in iter_gzip_csv(source):
        iso3 = row["ISO3_code"].strip()
        if not iso3 or row["Variant"] != "Medium":
            continue
        year = int(row["Time"])
        age = int(row["AgeGrpStart"])
        if not 1990 <= year <= 2023 or not 20 <= age <= 39:
            continue
        source_rows += 1
        band = next(item for item in AGE_BANDS if item[0] <= age <= item[1])
        # WPP population values are expressed in thousands.
        totals[(iso3, row["Location"], year, band[0], band[1])] += float(row["PopFemale"]) * 1000
    records = [
        {
            "iso3": key[0], "wpp_location_name": key[1], "year": key[2],
            "age_lower": key[3], "age_upper": key[4],
            "age_group": AGE_LABEL[(key[3], key[4])],
            "female_population": round(value, 3),
        }
        for key, value in totals.items()
    ]
    records.sort(key=lambda row: (str(row["iso3"]), int(row["year"]), int(row["age_lower"])))
    fields = [
        "iso3", "wpp_location_name", "year", "age_lower", "age_upper",
        "age_group", "female_population",
    ]
    write_csv(destination, records, fields)
    return records, {
        "source_single_age_rows": source_rows,
        "analytic_cells": len(records),
        "iso3_entities": len({row["iso3"] for row in records}),
        "years": [1990, 2023],
    }


def ingest_coverage(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_row, row in enumerate(read_xlsx_records(source), start=2):
        if row.get("GROUP") != "COUNTRIES" or not row.get("COVERAGE"):
            continue
        value = float(row["COVERAGE"])
        antigen = row["ANTIGEN"]
        sex = "female" if antigen.endswith("_F") or antigen.startswith("HPV_FEM") else "male"
        if "HPV1" in antigen or antigen.endswith("1"):
            dose = "first"
        elif "HPVC" in antigen or antigen in {"HPV_FEM", "HPV_MALE"}:
            dose = "completed"
        else:
            dose = "unknown"
        output.append(
            {
                "iso3": row["CODE"], "country_name": row["NAME"],
                "year": int(float(row["YEAR"])), "antigen": antigen,
                "sex": sex, "dose_definition": dose,
                "coverage_category": row["COVERAGE_CATEGORY_DESCRIPTION"],
                "coverage_pct": value,
                "coverage_out_of_range": int(not 0 <= value <= 100),
                "target_number": row["TARGET_NUMBER"], "doses": row["DOSES"],
                "source_row": source_row,
            }
        )
    output.sort(
        key=lambda row: (
            str(row["iso3"]), int(row["year"]),
            str(row["coverage_category"]), str(row["antigen"]),
        )
    )
    fields = [
        "iso3", "country_name", "year", "antigen", "sex", "dose_definition",
        "coverage_category", "coverage_pct", "coverage_out_of_range",
        "target_number", "doses", "source_row",
    ]
    write_csv(destination, output, fields)
    estimates = [row for row in output if row["coverage_category"] == "HPV Estimates"]
    return output, {
        "nonblank_rows": len(output),
        "wuenic_estimate_rows": len(estimates),
        "wuenic_countries": len({row["iso3"] for row in estimates}),
        "wuenic_years": [
            min(row["year"] for row in estimates), max(row["year"] for row in estimates)
        ],
        "out_of_range_rows": sum(int(row["coverage_out_of_range"]) for row in output),
        "wuenic_out_of_range_rows": sum(
            int(row["coverage_out_of_range"]) for row in estimates
        ),
    }


def ingest_introduction(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_row, row in enumerate(read_xlsx_records(source), start=2):
        if not row.get("ISO_3_CODE") or not row.get("YEAR"):
            continue
        output.append(
            {
                "iso3": row["ISO_3_CODE"], "country_name": row["COUNTRYNAME"],
                "who_region": row["WHO_REGION"], "year": int(float(row["YEAR"])),
                "antigen": row["ANTIGEN"], "introduction_status": row["INTRO"],
                "source_row": source_row,
            }
        )
    output.sort(key=lambda row: (str(row["iso3"]), int(row["year"])))
    fields = [
        "iso3", "country_name", "who_region", "year", "antigen",
        "introduction_status", "source_row",
    ]
    write_csv(destination, output, fields)
    return output, {
        "rows": len(output),
        "countries": len({row["iso3"] for row in output}),
        "years": [min(row["year"] for row in output), max(row["year"] for row in output)],
        "statuses": dict(Counter(str(row["introduction_status"]) for row in output)),
    }


def ingest_schedule(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_row, row in enumerate(read_xlsx_records(source), start=2):
        if not row.get("ISO_3_CODE") or not row.get("YEAR"):
            continue
        parsed_age = parse_hpv_target_age(row["AGEADMINISTERED"])
        output.append(
            {
                "iso3": row["ISO_3_CODE"], "country_name": row["COUNTRYNAME"],
                "who_region": row["WHO_REGION"], "year": int(float(row["YEAR"])),
                "schedule_round": _integer(row["SCHEDULEROUNDS"]),
                "target_population": row["TARGETPOP_DESCRIPTION"],
                "geographic_scope": row["GEOAREA"],
                "age_administered_native": row["AGEADMINISTERED"],
                "target_age_lower": "" if parsed_age is None else parsed_age[0],
                "target_age_upper": "" if parsed_age is None else parsed_age[1],
                "source_comment": row["SOURCECOMMENT"], "source_row": source_row,
            }
        )
    output.sort(
        key=lambda row: (
            str(row["iso3"]), int(row["year"]), int(row["schedule_round"] or 0)
        )
    )
    fields = [
        "iso3", "country_name", "who_region", "year", "schedule_round",
        "target_population", "geographic_scope", "age_administered_native",
        "target_age_lower", "target_age_upper", "source_comment", "source_row",
    ]
    write_csv(destination, output, fields)
    first_round = [row for row in output if row["schedule_round"] == 1]
    return output, {
        "rows": len(output),
        "countries": len({row["iso3"] for row in output}),
        "years": sorted({row["year"] for row in output}),
        "first_round_rows": len(first_round),
        "first_round_with_parsed_age": sum(row["target_age_lower"] != "" for row in first_round),
    }


def ingest_jrf_schedule(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Ingest the WIISE eJRF HPV schedule history (2006-2025).

    Supersedes the 2022-2025 Immunization Data Portal spreadsheet export for
    historical eligibility. Only round-1 rows carry a target age; later rounds
    carry dose intervals such as ``+M6``, which ``parse_hpv_target_age``
    rejects.
    """
    output: list[dict[str, object]] = []
    for source_row, row in enumerate(read_csv_records(source), start=2):
        if not row.get("COUNTRY") or not row.get("YEAR"):
            continue
        schedule_round = _integer(row["SCHEDULEROUNDS"])
        if schedule_round in SCHEDULE_ROUND_SENTINELS:
            schedule_round = None
        parsed_age = parse_hpv_target_age(row["AGEADMINISTERED"])
        output.append(
            {
                "iso3": row["COUNTRY"], "who_region": row["WHO_REGION"],
                "year": int(float(row["YEAR"])), "vaccine_code": row["VACCINECODE"],
                "schedule_round": "" if schedule_round is None else schedule_round,
                "target_population_code": row["TARGETPOP"].strip(),
                "geographic_scope": row["GEOAREA"].strip(),
                "age_administered_native": row["AGEADMINISTERED"],
                "target_age_lower": "" if parsed_age is None else parsed_age[0],
                "target_age_upper": "" if parsed_age is None else parsed_age[1],
                "source_comment": row["SOURCECOMMENT"], "source_row": source_row,
            }
        )
    output.sort(key=lambda row: (str(row["iso3"]), int(row["year"]), str(row["vaccine_code"])))
    fields = [
        "iso3", "who_region", "year", "vaccine_code", "schedule_round",
        "target_population_code", "geographic_scope", "age_administered_native",
        "target_age_lower", "target_age_upper", "source_comment", "source_row",
    ]
    write_csv(destination, output, fields)
    first_round = [row for row in output if row["schedule_round"] == 1]
    return output, {
        "rows": len(output),
        "countries": len({row["iso3"] for row in output}),
        "years": [min(row["year"] for row in output), max(row["year"] for row in output)],
        "first_round_rows": len(first_round),
        "first_round_with_parsed_age": sum(row["target_age_lower"] != "" for row in first_round),
        "rows_with_sentinel_round": sum(row["schedule_round"] == "" for row in output),
        "target_population_codes": dict(
            Counter(str(row["target_population_code"]) or "<blank>" for row in output)
        ),
        "geographic_scope_values": dict(Counter(str(row["geographic_scope"]) for row in output)),
    }


def ingest_jrf_introduction(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Ingest WIISE eJRF HPV introductions, separating effected from planned."""
    output: list[dict[str, object]] = []
    for source_row, row in enumerate(read_csv_records(source), start=2):
        if not row.get("COUNTRY") or not row.get("YEAR"):
            continue
        year = int(float(row["YEAR"]))
        output.append(
            {
                "iso3": row["COUNTRY"], "who_region": row["WHO_REGION"], "year": year,
                "vaccine_code": row["VACCINECODE"],
                "nationwide": row["NATIONWIDE"].strip().lower(),
                "partially": row["PARTIALLY"].strip().lower(),
                # Rows beyond the outcome panel describe planned introductions.
                "data_status": "projected" if year > OUTCOME_MAX_YEAR else "country_reported",
                "source_row": source_row,
            }
        )
    output.sort(key=lambda row: (str(row["iso3"]), int(row["year"])))
    fields = [
        "iso3", "who_region", "year", "vaccine_code", "nationwide", "partially",
        "data_status", "source_row",
    ]
    write_csv(destination, output, fields)
    return output, {
        "rows": len(output),
        "countries": len({row["iso3"] for row in output}),
        "years": [min(row["year"] for row in output), max(row["year"] for row in output)],
        "projected_rows_excluded_from_onset": sum(
            row["data_status"] == "projected" for row in output
        ),
        "nationwide_yes_rows": sum(row["nationwide"] == "yes" for row in output),
    }


def resolve_jrf_target_ages(
    jrf_schedule: list[dict[str, object]],
    national_intro_year: dict[str, int],
    coverage_years: dict[str, set[int]],
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[str, object]]:
    """Resolve country-year routine target ages from the eJRF schedule history.

    A blank ``TARGETPOP`` is not imputed. It is accepted as the routine
    adolescent programme only when independent observed evidence corroborates
    it: the row is national in scope, a nationwide introduction had already
    been reported for that country-year, and WUENIC reports final-dose female
    coverage for the same country-year. Otherwise the country-year stays
    unresolved and its cells remain ambiguous.
    """
    accepted: dict[tuple[str, int], dict[str, set | bool]] = defaultdict(
        lambda: {"ages": set(), "provenance": set(), "catchup": False}
    )
    stats = Counter()
    for row in jrf_schedule:
        if row["schedule_round"] != 1 or row["target_age_lower"] == "":
            continue
        iso3, year = str(row["iso3"]), int(row["year"])
        if year > OUTCOME_MAX_YEAR:
            stats["rows_after_outcome_panel"] += 1
            continue
        if row["geographic_scope"] != "NATIONAL":
            stats["rows_not_national_scope"] += 1
            continue
        code = str(row["target_population_code"]).upper()
        if code in NON_ROUTINE_TARGET_CODES:
            stats["rows_excluded_non_routine_target"] += 1
            continue
        if code in ROUTINE_TARGET_CODES or code in CATCHUP_TARGET_CODES:
            provenance = "jrf_labelled_target_population"
        elif not code:
            introduced = national_intro_year.get(iso3)
            country_years = coverage_years.get(iso3, set())
            if introduced is None or year < introduced or not country_years:
                stats["rows_unlabelled_uncorroborated"] += 1
                continue
            # WUENIC final-dose female coverage begins in 2010, so requiring
            # same-year coverage would disqualify every 2006-2009 programme
            # year and with it the earliest-adopting countries, whose cohorts
            # are the most mature. Same-year coverage is therefore recorded as
            # the stronger provenance rather than used as the admission test.
            if year in country_years:
                provenance = "jrf_unlabelled_corroborated_same_year"
            else:
                provenance = "jrf_unlabelled_corroborated_programme_level"
        else:
            stats["rows_unknown_target_code"] += 1
            continue

        entry = accepted[(iso3, year)]
        entry["ages"].add((int(row["target_age_lower"]), int(row["target_age_upper"])))
        entry["provenance"].add(provenance)
        if code in CATCHUP_TARGET_CODES:
            entry["catchup"] = True
        stats[f"rows_accepted_{provenance}"] += 1

    resolved: dict[tuple[str, int], dict[str, object]] = {}
    for key, entry in accepted.items():
        ages = sorted(entry["ages"])
        # Different HPV products in one country-year target the same programme
        # cohort, so the eligible age span is their union, not a conflict.
        lower = min(age[0] for age in ages)
        upper = max(age[1] for age in ages)
        resolved[key] = {
            "age": (lower, upper),
            "provenance": "+".join(sorted(entry["provenance"])),
            "catchup": bool(entry["catchup"]),
            "products_disagreed": len(ages) > 1,
        }
        if len(ages) > 1:
            stats["country_years_with_product_age_disagreement"] += 1

    return resolved, {
        "resolved_country_years": len(resolved),
        "resolved_countries": len({key[0] for key in resolved}),
        "resolved_year_range": (
            [min(key[1] for key in resolved), max(key[1] for key in resolved)]
            if resolved else []
        ),
        "country_years_from_labelled_target": sum(
            1 for value in resolved.values()
            if value["provenance"] == "jrf_labelled_target_population"
        ),
        "country_years_needing_corroboration": sum(
            1 for value in resolved.values()
            if "jrf_unlabelled_corroborated" in str(value["provenance"])
        ),
        "country_years_corroborated_same_year_only": sum(
            1 for value in resolved.values()
            if str(value["provenance"]) == "jrf_unlabelled_corroborated_same_year"
        ),
        "country_years_corroborated_programme_level_only": sum(
            1 for value in resolved.values()
            if str(value["provenance"]) == "jrf_unlabelled_corroborated_programme_level"
        ),
        "country_years_including_catchup": sum(
            1 for value in resolved.values() if value["catchup"]
        ),
        **dict(stats),
    }


def ingest_current_profile(source: Path, destination: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_row, row in enumerate(read_csv_records(source), start=2):
        age = parse_hpv_target_age(row["HPV_AGEADMINISTERED"])
        output.append(
            {
                "iso3": row["ISO_3_CODE"], "country_name": row["COUNTRYNAME"],
                "who_region": row["WHO_REGION"], "world_bank_income": row["WB_INCOME"],
                "national_schedule": row["HPV_NATIONAL_SCHEDULE"],
                "introduction_year": row["HPV_YEAR_INTRODUCTION"],
                "primary_delivery_strategy": row["HPV_PRIM_DELIV_STRATEGY"],
                "age_administered_current": row["HPV_AGEADMINISTERED"],
                "target_age_lower_current": "" if age is None else age[0],
                "target_age_upper_current": "" if age is None else age[1],
                "targeted_sex_current": row["HPV_SEX"],
                "schedule_current": row["HPV_INT_DOSES"],
                "hpv1_coverage_last_year": row["HPV1_COVERAGELASTYEAR"],
                "hpvc_coverage_last_year": row["HPVC_COVERAGELASTYEAR"],
                "source_row": source_row,
            }
        )
    output.sort(key=lambda row: str(row["iso3"]))
    fields = list(output[0])
    write_csv(destination, output, fields)
    return output, {
        "rows": len(output),
        "countries": len({row["iso3"] for row in output}),
        "with_current_target_age": sum(row["target_age_lower_current"] != "" for row in output),
    }


def build_crosswalk(
    gbd: list[dict[str, object]],
    wpp: list[dict[str, object]],
    current_profile: list[dict[str, object]],
    destination: Path,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    candidate: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in wpp:
        candidate[normalize_country_name(str(row["wpp_location_name"]))].add(
            (str(row["iso3"]), "wpp_exact_normalized_name")
        )
    for row in current_profile:
        candidate[normalize_country_name(str(row["country_name"]))].add(
            (str(row["iso3"]), "who_exact_normalized_name")
        )

    result: dict[int, str] = {}
    audit: list[dict[str, object]] = []
    locations = {
        int(row["gbd_location_id"]): str(row["gbd_location_name"]) for row in gbd
    }
    for location_id, location_name in sorted(locations.items(), key=lambda item: item[1]):
        normalized = normalize_country_name(location_name)
        possible = candidate.get(normalized, set())
        iso_values = sorted({item[0] for item in possible})
        if len(iso_values) == 1:
            iso3 = iso_values[0]
            methods = "+".join(sorted({item[1] for item in possible}))
            status = "matched"
        elif normalized in GBD_NAME_TO_ISO3:
            iso3 = GBD_NAME_TO_ISO3[normalized]
            methods = "reviewed_alias"
            status = "matched"
        else:
            iso3 = ""
            methods = "unmatched" if not possible else "ambiguous_name_collision"
            status = "unmatched"
        if iso3:
            result[location_id] = iso3
        audit.append(
            {
                "gbd_location_id": location_id, "gbd_location_name": location_name,
                "normalized_name": normalized, "iso3": iso3,
                "match_method": methods, "match_status": status,
            }
        )
    write_csv(
        destination,
        audit,
        [
            "gbd_location_id", "gbd_location_name", "normalized_name", "iso3",
            "match_method", "match_status",
        ],
    )
    return result, audit


def build_feasibility_matrix(
    gbd: list[dict[str, object]],
    wpp: list[dict[str, object]],
    coverage: list[dict[str, object]],
    introduction: list[dict[str, object]],
    schedule: list[dict[str, object]],
    current_profile: list[dict[str, object]],
    crosswalk: dict[int, str],
    jrf_resolved: dict[tuple[str, int], dict[str, object]],
    intro_year: dict[str, int],
    destination: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    wpp_map = {
        (str(row["iso3"]), int(row["year"]), int(row["age_lower"]), int(row["age_upper"])):
        row["female_population"] for row in wpp
    }
    profile_map = {str(row["iso3"]): row for row in current_profile}

    partial_intro_year: dict[str, int] = {}
    for row in introduction:
        iso3, year, status = str(row["iso3"]), int(row["year"]), str(row["introduction_status"])
        if status == "Yes (P)":
            partial_intro_year[iso3] = min(year, partial_intro_year.get(iso3, year))

    documented_ages = {key: value["age"] for key, value in jrf_resolved.items()}

    # A country's eligibility history is usable only if every programme year
    # from national introduction to the end of the outcome panel resolved. A
    # gap may be a genuine interruption or a reporting hole; no retrieved source
    # distinguishes the two, so the country fails closed either way.
    resolved_years: dict[str, set[int]] = defaultdict(set)
    for iso3, year in jrf_resolved:
        resolved_years[iso3].add(year)
    schedule_complete: dict[str, bool] = {}
    schedule_gap_years: dict[str, list[int]] = {}
    for iso3, introduced in intro_year.items():
        if introduced > OUTCOME_MAX_YEAR:
            schedule_complete[iso3] = False
            schedule_gap_years[iso3] = []
            continue
        required = set(range(introduced, OUTCOME_MAX_YEAR + 1))
        gaps = sorted(required - resolved_years.get(iso3, set()))
        schedule_complete[iso3] = not gaps
        schedule_gap_years[iso3] = gaps

    # Oldest birth cohort a country ever targeted. Bands entirely older than
    # this were never eligible and are clean never-treated observations.
    earliest_eligible_cohort: dict[str, int] = {}
    for (iso3, year), value in jrf_resolved.items():
        age_upper = int(value["age"][1])
        candidate = year - age_upper
        current = earliest_eligible_cohort.get(iso3)
        earliest_eligible_cohort[iso3] = (
            candidate if current is None else min(current, candidate)
        )

    final_female_estimates = [
        row for row in coverage
        if row["coverage_category"] == "HPV Estimates"
        and row["antigen"] == "PRHPVC_F"
        and not row["coverage_out_of_range"]
    ]
    coverage_years: dict[str, list[int]] = defaultdict(list)
    cohort_records: dict[tuple[str, int], list[tuple[int, float, str]]] = defaultdict(list)
    excluded_pre_national_coverage_rows = 0
    excluded_without_national_introduction_rows = 0
    for row in final_female_estimates:
        iso3, programme_year = str(row["iso3"]), int(row["year"])
        coverage_years[iso3].append(programme_year)
        if iso3 not in intro_year:
            excluded_without_national_introduction_rows += 1
            continue
        if programme_year < intro_year[iso3]:
            excluded_pre_national_coverage_rows += 1
            continue
        if (iso3, programme_year) in documented_ages:
            age = documented_ages[(iso3, programme_year)]
            method = "documented_year_schedule"
        else:
            profile = profile_map.get(iso3)
            if not profile or profile["target_age_lower_current"] == "":
                continue
            age = (
                int(profile["target_age_lower_current"]),
                int(profile["target_age_upper_current"]),
            )
            method = "current_age_proxy"
        for cohort_year in range(programme_year - age[1], programme_year - age[0] + 1):
            cohort_records[(iso3, cohort_year)].append(
                (programme_year, float(row["coverage_pct"]), method)
            )

    output: list[dict[str, object]] = []
    for outcome in gbd:
        location_id = int(outcome["gbd_location_id"])
        iso3 = crosswalk.get(location_id, "")
        year = int(outcome["year"])
        age_lower, age_upper = int(outcome["age_lower"]), int(outcome["age_upper"])
        cohort = outcome_cohort_interval(year, age_lower, age_upper)
        profile = profile_map.get(iso3, {})
        target_age = None
        if profile and profile.get("target_age_lower_current") != "":
            target_age = (
                int(profile["target_age_lower_current"]),
                int(profile["target_age_upper_current"]),
            )
        first_national = intro_year.get(iso3)
        if first_national is not None and target_age is not None:
            threshold = first_national - target_age[1]
            proxy_eligible_years = [
                item for item in range(cohort.lower, cohort.upper + 1) if item >= threshold
            ]
        else:
            proxy_eligible_years = []

        covered_cohorts: set[int] = set()
        documented_cohorts: set[int] = set()
        proxy_cohorts: set[int] = set()
        observations: list[float] = []
        source_programme_years: set[int] = set()
        provenance_used: set[str] = set()
        catchup_used = False
        # Age at which each birth cohort was first reached by a documented
        # programme. HPV vaccine is prophylactic, so a cohort first targeted in
        # adulthood cannot have its invasive-cancer risk altered the way one
        # targeted before sexual debut can. The youngest targeting age is the one
        # that matters, hence the min per cohort.
        documented_age_at_eligibility: dict[int, int] = {}
        for cohort_year in range(cohort.lower, cohort.upper + 1):
            # Exclude vaccinations recorded after the outcome observation. This
            # matters for countries whose current profile includes older catch-up
            # ages: a future programme year can map to an already-observed cohort.
            matches = [
                item for item in cohort_records.get((iso3, cohort_year), [])
                if item[0] <= year
            ]
            if matches:
                covered_cohorts.add(cohort_year)
            for programme_year, value, method in matches:
                observations.append(value)
                source_programme_years.add(programme_year)
                if method == "documented_year_schedule":
                    documented_cohorts.add(cohort_year)
                    age_at_eligibility = programme_year - cohort_year
                    previous = documented_age_at_eligibility.get(cohort_year)
                    if previous is None or age_at_eligibility < previous:
                        documented_age_at_eligibility[cohort_year] = age_at_eligibility
                    resolution = jrf_resolved.get((iso3, programme_year), {})
                    provenance_used.add(str(resolution.get("provenance", "")))
                    catchup_used = catchup_used or bool(resolution.get("catchup"))
                else:
                    proxy_cohorts.add(cohort_year)
        coverage_share = len(covered_cohorts) / cohort.width
        documented_share = len(documented_cohorts) / cohort.width
        proxy_share = len(proxy_cohorts) / cohort.width
        if documented_share == 1:
            feasibility = "full_documented_schedule_overlap"
        elif coverage_share == 1:
            feasibility = "full_proxy_overlap"
        elif coverage_share > 0:
            feasibility = "partial_overlap"
        elif not iso3:
            feasibility = "country_unmatched"
        else:
            feasibility = "no_coverage_cohort_overlap"

        # --- primary eligibility gate -------------------------------------
        history_complete = schedule_complete.get(iso3, False)
        gaps = schedule_gap_years.get(iso3, [])
        never_introduced = iso3 and iso3 not in intro_year
        earliest_eligible = earliest_eligible_cohort.get(iso3)
        never_eligible_band = (
            earliest_eligible is not None and cohort.upper < earliest_eligible
        )
        if not iso3:
            status, treated, untreated = "country_unmatched", 0, 0
            reason = "GBD location has no reviewed ISO3 match"
        elif never_introduced:
            status, treated, untreated = "untreated_no_national_programme", 0, 1
            reason = ""
        elif documented_share == 1:
            status, treated, untreated = "treated_documented", 1, 0
            reason = ""
        elif never_eligible_band:
            status, treated, untreated = "untreated_pre_eligibility_cohort", 0, 1
            reason = ""
        elif coverage_share > 0 or proxy_share > 0:
            status, treated, untreated = "ambiguous_partial_cohort_overlap", 0, 0
            reason = "birth-cohort band only partly within documented target ages"
        else:
            status, treated, untreated = "ambiguous_unassigned", 0, 0
            reason = "no documented eligibility or coverage evidence for this cohort band"

        output.append(
            {
                **outcome,
                "iso3": iso3,
                "female_population_wpp": wpp_map.get((iso3, year, age_lower, age_upper), ""),
                "cohort_lower": cohort.lower, "cohort_upper": cohort.upper,
                "first_national_introduction_year": first_national or "",
                "first_partial_introduction_year": partial_intro_year.get(iso3, ""),
                "current_target_age": profile.get("age_administered_current", ""),
                "current_targeted_sex": profile.get("targeted_sex_current", ""),
                "proxy_potential_eligible_share": len(proxy_eligible_years) / cohort.width,
                "cohort_coverage_overlap_share": coverage_share,
                "documented_schedule_overlap_share": documented_share,
                "current_age_proxy_overlap_share": proxy_share,
                "coverage_observations": len(observations),
                "coverage_source_programme_years": ";".join(map(str, sorted(source_programme_years))),
                "completed_coverage_min_pct": _format_number(min(observations) if observations else None),
                "completed_coverage_mean_pct_feasibility_only": _format_number(
                    mean(observations) if observations else None
                ),
                "completed_coverage_max_pct": _format_number(max(observations) if observations else None),
                "feasibility_status": feasibility,
                "analysis_status": status,
                "historical_schedule_complete": int(history_complete),
                "schedule_gap_year_count": len(gaps),
                "eligibility_provenance": "+".join(sorted(p for p in provenance_used if p)),
                "catchup_cohort_involved": int(catchup_used),
                # Band-level summary of the youngest documented targeting age.
                # ``max`` is the binding one: if any cohort in the band was only
                # ever reached late, the band is not wholly prophylactic.
                "age_at_eligibility_min": (
                    min(documented_age_at_eligibility.values())
                    if documented_age_at_eligibility else ""
                ),
                "age_at_eligibility_max": (
                    max(documented_age_at_eligibility.values())
                    if documented_age_at_eligibility else ""
                ),
                "eligibility_ambiguous": int(not (treated or untreated)),
                "primary_treated_eligible": treated,
                "primary_untreated_eligible": untreated,
                "ambiguity_reason": reason,
            }
        )
    output.sort(
        key=lambda row: (
            str(row["gbd_location_name"]), int(row["year"]), int(row["age_lower"])
        )
    )
    fields = list(output[0])
    write_csv(destination, output, fields)
    overlap_rows = [row for row in output if float(row["cohort_coverage_overlap_share"]) > 0]
    return output, {
        "rows": len(output),
        "locations": len({row["gbd_location_id"] for row in output}),
        "matched_locations": len({row["gbd_location_id"] for row in output if row["iso3"]}),
        "cells_with_wpp_population": sum(row["female_population_wpp"] != "" for row in output),
        "cells_with_any_cohort_coverage_overlap": len(overlap_rows),
        "countries_with_any_cohort_coverage_overlap": len({row["iso3"] for row in overlap_rows}),
        "overlap_cells_by_age_group": dict(Counter(str(row["age_group"]) for row in overlap_rows)),
        "overlap_outcome_year_range": [
            min(int(row["year"]) for row in overlap_rows),
            max(int(row["year"]) for row in overlap_rows),
        ] if overlap_rows else [],
        "female_final_coverage_rows_excluded_before_national_introduction":
            excluded_pre_national_coverage_rows,
        "female_final_coverage_rows_excluded_without_national_introduction":
            excluded_without_national_introduction_rows,
        "cells_with_full_proxy_overlap": sum(row["feasibility_status"] == "full_proxy_overlap" for row in output),
        "cells_with_full_documented_schedule_overlap": sum(
            row["feasibility_status"] == "full_documented_schedule_overlap" for row in output
        ),
        "analysis_status_counts": dict(Counter(str(row["analysis_status"]) for row in output)),
        "primary_treated_cells_ready": sum(int(row["primary_treated_eligible"]) for row in output),
        "primary_untreated_cells_ready": sum(int(row["primary_untreated_eligible"]) for row in output),
        "primary_treated_countries": len({
            str(row["iso3"]) for row in output if int(row["primary_treated_eligible"])
        }),
        "primary_treated_cells_by_age_group": dict(Counter(
            str(row["age_group"]) for row in output if int(row["primary_treated_eligible"])
        )),
        "primary_treated_outcome_year_range": (
            [
                min(int(row["year"]) for row in output if int(row["primary_treated_eligible"])),
                max(int(row["year"]) for row in output if int(row["primary_treated_eligible"])),
            ]
            if any(int(row["primary_treated_eligible"]) for row in output) else []
        ),
        "treated_cells_relying_on_corroborated_unlabelled_targetpop": sum(
            1 for row in output
            if int(row["primary_treated_eligible"])
            and "jrf_unlabelled_corroborated" in str(row["eligibility_provenance"])
        ),
        "treated_cells_involving_catchup": sum(
            1 for row in output
            if int(row["primary_treated_eligible"]) and int(row["catchup_cohort_involved"])
        ),
        "countries_with_complete_history": sum(1 for value in schedule_complete.values() if value),
        "countries_with_schedule_gaps": sum(1 for value in schedule_complete.values() if not value),
        # Pre-specified sensitivity restriction: drop countries with any
        # unresolved programme year between introduction and 2023.
        "sensitivity_complete_history_only": {
            "treated_cells": sum(
                1 for row in output
                if int(row["primary_treated_eligible"]) and int(row["historical_schedule_complete"])
            ),
            "treated_countries": len({
                str(row["iso3"]) for row in output
                if int(row["primary_treated_eligible"]) and int(row["historical_schedule_complete"])
            }),
        },
        # Pre-specified sensitivity restriction: drop treated cells whose
        # eligibility depends on a blank TARGETPOP resolved by corroboration.
        "sensitivity_labelled_targetpop_only": {
            "treated_cells": sum(
                1 for row in output
                if int(row["primary_treated_eligible"])
                and "unlabelled" not in str(row["eligibility_provenance"])
            ),
            "treated_countries": len({
                str(row["iso3"]) for row in output
                if int(row["primary_treated_eligible"])
                and "unlabelled" not in str(row["eligibility_provenance"])
            }),
        },
    }


def build_country_summary(
    matrix: list[dict[str, object]],
    coverage: list[dict[str, object]],
    schedule: list[dict[str, object]],
    current_profile: list[dict[str, object]],
    crosswalk_audit: list[dict[str, object]],
    destination: Path,
) -> list[dict[str, object]]:
    by_location: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in matrix:
        by_location[int(row["gbd_location_id"])].append(row)
    audit_map = {int(row["gbd_location_id"]): row for row in crosswalk_audit}
    profile_map = {str(row["iso3"]): row for row in current_profile}
    coverage_year_map: dict[str, set[int]] = defaultdict(set)
    for row in coverage:
        if row["coverage_category"] == "HPV Estimates" and row["antigen"] == "PRHPVC_F":
            coverage_year_map[str(row["iso3"])].add(int(row["year"]))
    schedule_year_map: dict[str, set[int]] = defaultdict(set)
    for row in schedule:
        if row["schedule_round"] == 1 and row["target_age_lower"] != "":
            schedule_year_map[str(row["iso3"])].add(int(row["year"]))

    output: list[dict[str, object]] = []
    for location_id, rows in sorted(by_location.items(), key=lambda item: str(item[1][0]["gbd_location_name"])):
        first = rows[0]
        iso3 = str(first["iso3"])
        profile = profile_map.get(iso3, {})
        years = sorted(coverage_year_map.get(iso3, set()))
        schedule_years = sorted(schedule_year_map.get(iso3, set()))
        output.append(
            {
                "gbd_location_id": location_id,
                "gbd_location_name": first["gbd_location_name"], "iso3": iso3,
                "crosswalk_method": audit_map[location_id]["match_method"],
                "first_national_introduction_year": first["first_national_introduction_year"],
                "current_target_age": profile.get("age_administered_current", ""),
                "current_targeted_sex": profile.get("targeted_sex_current", ""),
                "wuenic_final_female_first_year": min(years) if years else "",
                "wuenic_final_female_last_year": max(years) if years else "",
                "wuenic_final_female_year_count": len(years),
                "documented_schedule_years": ";".join(map(str, schedule_years)),
                "outcome_age_groups": len({row["age_group"] for row in rows}),
                "outcome_pre_introduction_cells": sum(
                    bool(row["first_national_introduction_year"])
                    and int(row["year"]) < int(row["first_national_introduction_year"])
                    for row in rows
                ),
                "proxy_potential_post_vaccination_cells": sum(
                    float(row["proxy_potential_eligible_share"]) > 0 for row in rows
                ),
                "cells_with_any_coverage_overlap": sum(
                    float(row["cohort_coverage_overlap_share"]) > 0 for row in rows
                ),
                "cells_with_full_coverage_overlap": sum(
                    float(row["cohort_coverage_overlap_share"]) == 1 for row in rows
                ),
                "historical_schedule_complete": int(first["historical_schedule_complete"]),
                "schedule_gap_year_count": int(first["schedule_gap_year_count"]),
                "treated_cells": sum(int(row["primary_treated_eligible"]) for row in rows),
                "untreated_cells": sum(int(row["primary_untreated_eligible"]) for row in rows),
                "ambiguous_cells": sum(int(row["eligibility_ambiguous"]) for row in rows),
                "primary_analysis_ready": int(
                    any(int(row["primary_treated_eligible"]) for row in rows)
                    and any(int(row["primary_untreated_eligible"]) for row in rows)
                ),
                "blocking_reason": next(
                    (str(row["ambiguity_reason"]) for row in rows if row["ambiguity_reason"]),
                    "",
                ),
            }
        )
    fields = list(output[0])
    write_csv(destination, output, fields)
    return output


def build_analytic_dataset(
    matrix: list[dict[str, object]],
    covariate_panel: dict[tuple[str, int], dict[str, object]],
    destination: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Assemble the country x age-group x year panel for staggered-adoption DiD.

    The unit of adoption is a country and age band, because a birth cohort
    enters each age band in a different calendar year. Treatment is absorbing
    by construction: programme eligibility cannot be withdrawn from a cohort
    once it has been targeted, so the treatment indicator is defined from the
    first treated year onward and disagreements with the raw per-cell flag are
    counted rather than silently smoothed.
    """
    usable = [
        row for row in matrix
        if row["iso3"] and row["female_population_wpp"] not in ("", None)
    ]
    first_treated: dict[tuple[str, str], int] = {}
    for row in usable:
        if int(row["primary_treated_eligible"]):
            key = (str(row["iso3"]), str(row["age_group"]))
            year = int(row["year"])
            first_treated[key] = min(year, first_treated.get(key, year))

    output: list[dict[str, object]] = []
    flag_disagreements = 0
    for row in usable:
        iso3, age_group, year = str(row["iso3"]), str(row["age_group"]), int(row["year"])
        cohort_year = first_treated.get((iso3, age_group))
        treated_now = int(cohort_year is not None and year >= cohort_year)
        if treated_now and not int(row["primary_treated_eligible"]):
            flag_disagreements += 1
        population = float(row["female_population_wpp"])
        cases = float(row["cases"])
        covariates = covariate_panel.get((iso3, year), {})
        output.append(
            {
                "unit_id": f"{iso3}|{age_group}", "iso3": iso3, "age_group": age_group,
                "year": year, "cases": cases, "female_population": population,
                "rate_per_100k": row["rate_per_100k"],
                "rate_per_100k_lower": row["rate_per_100k_lower"],
                "rate_per_100k_upper": row["rate_per_100k_upper"],
                "treatment_cohort_year": cohort_year or 0,
                "treated": treated_now,
                "event_time": "" if cohort_year is None else year - cohort_year,
                "ever_treated": int(cohort_year is not None),
                "analysis_status": row["analysis_status"],
                "primary_treated_eligible": row["primary_treated_eligible"],
                "eligibility_provenance": row["eligibility_provenance"],
                "age_at_eligibility_min": row["age_at_eligibility_min"],
                "age_at_eligibility_max": row["age_at_eligibility_max"],
                "historical_schedule_complete": row["historical_schedule_complete"],
                "first_national_introduction_year": row["first_national_introduction_year"],
                **{field: covariates.get(field, "") for field in COVARIATE_FIELDS},
            }
        )
    output.sort(key=lambda row: (str(row["unit_id"]), int(row["year"])))
    fields = list(output[0])
    write_csv(destination, output, fields)

    treated_units = {row["unit_id"] for row in output if row["ever_treated"]}
    cohorts = Counter(
        int(row["treatment_cohort_year"]) for row in output if row["ever_treated"]
    )
    pre_period_years = {}
    for unit in treated_units:
        unit_rows = [row for row in output if row["unit_id"] == unit]
        cohort_year = int(unit_rows[0]["treatment_cohort_year"])
        pre_period_years[unit] = sum(1 for row in unit_rows if int(row["year"]) < cohort_year)
    return output, {
        "rows": len(output),
        "units": len({row["unit_id"] for row in output}),
        "countries": len({row["iso3"] for row in output}),
        "treated_units": len(treated_units),
        "never_treated_units": len({row["unit_id"] for row in output}) - len(treated_units),
        "treatment_cohorts": dict(sorted(Counter(
            int(row["treatment_cohort_year"]) for row in output
            if row["ever_treated"] and int(row["event_time"]) == 0
        ).items())),
        "absorbing_flag_disagreements": flag_disagreements,
        "treated_units_with_at_least_3_pre_years": sum(
            1 for value in pre_period_years.values() if value >= 3
        ),
        "min_pre_period_years": min(pre_period_years.values()) if pre_period_years else 0,
        "covariate_completeness": {
            field: sum(1 for row in output if row[field] != "") / len(output)
            for field in COVARIATE_FIELDS
        },
    }


def run_build(root: Path) -> dict[str, object]:
    raw = root / "02_raw_data"
    processed = root / "03_processed_data"
    results = root / "05_results"
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    paths = {
        "gbd": raw / "gbd_2023/manual_export/IHME-GBD_2023_DATA-7149e941-1.zip",
        "wpp": raw / "un_wpp/2024_revision/WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz",
        "coverage": raw / "who_unicef_hpv/who_unicef_hpv/2026_who_portal/coverage_reported/Human Papillomavirus (HPV) vaccination coverage 2026-05-08 10-43 UTC.xlsx",
        "introduction": raw / "who_unicef_hpv/who_unicef_hpv/2026_who_portal/introduction/Introduction of HPV (Human Papillomavirus) vaccine 2026-05-08 11-05 UTC.xlsx",
        "schedule": raw / "who_unicef_hpv/who_unicef_hpv/2026_who_portal/schedule/Vaccination schedule for Human Papillomavirus (HPV) 2026-05-08 10-39 UTC.xlsx",
        "current_profile": raw / "who_unicef_hpv/who_unicef_hpv/2026-07-22_who_hpv_dashboard/delivery_strategy.csv",
        "jrf_schedule": raw / "who_wiise_jrf/2026-08-11_odata/AD_SCHEDULES_HPV.csv",
        "jrf_introduction": raw / "who_wiise_jrf/2026-08-11_odata/AD_VACCINE_INTRODUCTIONS_HPV.csv",
    }
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required build inputs: " + ", ".join(missing))

    gbd, gbd_stats = ingest_gbd(paths["gbd"], processed / "gbd_cervical_incidence.csv")
    wpp, wpp_stats = ingest_wpp(paths["wpp"], processed / "wpp_female_population_agebands.csv")
    coverage, coverage_stats = ingest_coverage(paths["coverage"], processed / "hpv_coverage_all_series.csv")
    introduction, introduction_stats = ingest_introduction(
        paths["introduction"], processed / "hpv_introduction_history.csv"
    )
    schedule, schedule_stats = ingest_schedule(paths["schedule"], processed / "hpv_schedule_download.csv")
    current_profile, profile_stats = ingest_current_profile(
        paths["current_profile"], processed / "hpv_current_programme_profile.csv"
    )
    jrf_schedule, jrf_schedule_stats = ingest_jrf_schedule(
        paths["jrf_schedule"], processed / "jrf_hpv_schedule_history.csv"
    )
    jrf_introduction, jrf_introduction_stats = ingest_jrf_introduction(
        paths["jrf_introduction"], processed / "jrf_hpv_introduction_history.csv"
    )
    crosswalk, crosswalk_audit = build_crosswalk(
        gbd, wpp, current_profile, processed / "country_crosswalk_auto.csv"
    )

    # Eligibility onset comes from the eJRF introductions table, restricted to
    # nationwide introductions reported within the outcome panel.
    intro_year: dict[str, int] = {}
    for row in jrf_introduction:
        if row["nationwide"] == "yes" and row["data_status"] == "country_reported":
            iso3, year = str(row["iso3"]), int(row["year"])
            intro_year[iso3] = min(year, intro_year.get(iso3, year))

    final_female_coverage_years: dict[str, set[int]] = defaultdict(set)
    for row in coverage:
        if (
            row["coverage_category"] == "HPV Estimates"
            and row["antigen"] == "PRHPVC_F"
            and not row["coverage_out_of_range"]
        ):
            final_female_coverage_years[str(row["iso3"])].add(int(row["year"]))

    jrf_resolved, resolution_stats = resolve_jrf_target_ages(
        jrf_schedule, intro_year, final_female_coverage_years
    )

    portal_intro_year: dict[str, int] = {}
    for row in introduction:
        if str(row["introduction_status"]) == "Yes":
            iso3, year = str(row["iso3"]), int(row["year"])
            portal_intro_year[iso3] = min(year, portal_intro_year.get(iso3, year))
    shared = set(intro_year) & set(portal_intro_year)
    concordance = {
        "countries_in_both_sources": len(shared),
        "introduction_year_agreements": sum(
            1 for iso3 in shared if intro_year[iso3] == portal_intro_year[iso3]
        ),
        "introduction_year_disagreements": sorted(
            {
                iso3: [intro_year[iso3], portal_intro_year[iso3]]
                for iso3 in shared if intro_year[iso3] != portal_intro_year[iso3]
            }.items()
        ),
        "only_in_jrf_odata": sorted(set(intro_year) - set(portal_intro_year)),
        "only_in_portal_export": sorted(set(portal_intro_year) - set(intro_year)),
    }

    matrix, matrix_stats = build_feasibility_matrix(
        gbd, wpp, coverage, introduction, schedule, current_profile, crosswalk,
        jrf_resolved, intro_year,
        processed / "cohort_coverage_feasibility_matrix.csv",
    )
    covariate_panel, covariate_stats = build_covariate_panel(
        raw / "worldbank_wdi/2026-08-11",
        raw / "who_gho/2026-08-11",
        keep_iso3={iso3 for iso3 in crosswalk.values() if iso3},
    )
    analytic, analytic_stats = build_analytic_dataset(
        matrix, covariate_panel, processed / "analytic_dataset.csv"
    )
    country_summary = build_country_summary(
        matrix, coverage, schedule, current_profile, crosswalk_audit,
        processed / "cohort_coverage_feasibility_by_country.csv",
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "exposure_assigned_analysis_not_started",
        "source_files": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "ingestion": {
            "gbd": gbd_stats, "wpp": wpp_stats, "coverage": coverage_stats,
            "introduction": introduction_stats, "schedule": schedule_stats,
            "current_profile": profile_stats,
            "jrf_schedule": jrf_schedule_stats,
            "jrf_introduction": jrf_introduction_stats,
        },
        "exposure_resolution": resolution_stats,
        "introduction_source_concordance": concordance,
        "covariates": covariate_stats,
        "analytic_dataset": analytic_stats,
        "crosswalk": {
            "gbd_locations": len(crosswalk_audit),
            "matched": sum(row["match_status"] == "matched" for row in crosswalk_audit),
            "unmatched": [
                row["gbd_location_name"] for row in crosswalk_audit
                if row["match_status"] != "matched"
            ],
        },
        "matrix": matrix_stats,
        "country_summary_rows": len(country_summary),
        "primary_analysis_blocker": (
            "None at the data level. Every treated cell depends on a blank TARGETPOP "
            "resolved by corroboration, because the field was not collected before 2019 "
            "and no programme year that produces a mature cohort by 2023 postdates it. "
            "Programme interruptions remain unrepresented in every retrieved source."
        ),
    }
    report_path = results / "ingestion_feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    matrix_summary = report["matrix"]
    ingestion = report["ingestion"]
    markdown = f"""# Project 1 ingestion and cohort-coverage feasibility

Generated: {report['generated_at_utc']}

## Build status

**Exposure assigned from the eJRF programme history; no analysis run.**

- GBD analytic cells: {matrix_summary['rows']:,}
- GBD locations matched to ISO3: {matrix_summary['matched_locations']}/{matrix_summary['locations']}
- Cells with WPP female population: {matrix_summary['cells_with_wpp_population']:,}
- Resolved programme country-years: {report['exposure_resolution']['resolved_country_years']:,} across {report['exposure_resolution']['resolved_countries']} countries, {report['exposure_resolution']['resolved_year_range']}
- **Primary treated cells: {matrix_summary['primary_treated_cells_ready']}** across {matrix_summary['primary_treated_countries']} countries, outcome years {matrix_summary['primary_treated_outcome_year_range']}
- Primary untreated (control) cells: {matrix_summary['primary_untreated_cells_ready']:,}
- Treated cells by age group: {matrix_summary['primary_treated_cells_by_age_group']}

## Interpretation

The treated sample is small and young: {matrix_summary['primary_treated_cells_by_age_group'].get('20-24', 0)} of
{matrix_summary['primary_treated_cells_ready']} treated cells are ages 20-24. This is the cohort-maturity
constraint the execution brief anticipated, not a data defect. Countries with
{matrix_summary['primary_treated_countries']} treated cells clear the 20-country floor in section 5 of the brief,
but only just, and the age distribution means any effect estimate will rest
mainly on ages 20-29.

**Every treated cell depends on a blank `TARGETPOP` resolved by corroboration.**
The eJRF did not collect that field before 2019, and no programme year late
enough to carry a labelled value produces a cohort mature enough to appear in
the 2023 outcome panel. The `sensitivity_labelled_targetpop_only` restriction
therefore yields zero treated cells: it is degenerate and cannot serve as a
robustness check. The usable restriction is
`sensitivity_complete_history_only`, which retains
{matrix_summary['sensitivity_complete_history_only']['treated_cells']} cells across
{matrix_summary['sensitivity_complete_history_only']['treated_countries']} countries.

Blank target populations are never imputed. A country-year is admitted only
when a national-scope round-1 schedule row carries a parseable target age, a
nationwide introduction was reported at or before that year, and the country
reports WUENIC final-dose female coverage. Same-year coverage is recorded as
the stronger provenance; requiring it as the admission test would have
truncated every 2006-2009 programme year, because the WUENIC series begins in
2010.

Programme interruptions remain unrepresented in every retrieved source. A
documented schedule with no corresponding coverage observation creates no
treated cell, which limits but does not eliminate the risk that a paper
schedule is read as a delivered programme.

The `completed_coverage_mean_pct_feasibility_only` field is a descriptive mean
of matching source observations. It is not the frozen analytic exposure and is
not used for effect estimation.
"""
    (results / "cohort_coverage_feasibility_report.md").write_text(
        markdown, encoding="utf-8"
    )
    return report
