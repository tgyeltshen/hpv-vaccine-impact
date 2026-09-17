"""Single restartable entry point for Project 1.

Validation, feasibility-data construction and effect estimation are enabled.
The analysis stage runs the group-time estimators and then applies the
protocol's stop rules to its own output; it exits non-zero when those rules
forbid a causal conclusion, so a green pipeline never implies a positive
finding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    path = ROOT / "01_metadata" / "source_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def workbook_sheet_names(path: Path) -> list[str]:
    """Read XLSX sheet names using the standard library, without altering bytes."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("xl/workbook.xml")
    root = ET.fromstring(xml)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [element.attrib["name"] for element in root.findall("m:sheets/m:sheet", ns)]


def validate() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    manifest = read_manifest()
    by_path = {row["local_path"]: row for row in manifest}

    for relative, row in by_path.items():
        path = ROOT / relative
        if not path.exists():
            failures.append(f"Manifested file missing: {relative}")
            continue
        actual = file_sha256(path)
        if actual != row["sha256"]:
            failures.append(f"Checksum mismatch: {relative}")

    hpv_relative = "02_raw_data/who_unicef_hpv/2025_revision/hpv2025rev_web-update.xlsx"
    hpv_path = ROOT / hpv_relative
    hpv_sheets: list[str] = []
    if not hpv_path.exists():
        failures.append("WHO/UNICEF HPV coverage workbook not downloaded")
    else:
        try:
            hpv_sheets = workbook_sheet_names(hpv_path)
        except Exception as exc:
            failures.append(f"HPV workbook unreadable as XLSX: {exc}")

    gbd_dir = ROOT / "02_raw_data" / "gbd_2023" / "manual_export"
    gbd_files = sorted(
        p for p in gbd_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not gbd_files:
        failures.append("BLOCKER: primary GBD 2023 outcome export is absent")
    else:
        unmanifested = [p.name for p in gbd_files if p.relative_to(ROOT).as_posix() not in by_path]
        if unmanifested:
            warnings.append(
                "Manual GBD files require manifest registration before parsing: "
                + ", ".join(unmanifested)
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "validate",
        "status": "blocked" if failures else "ready",
        "failures": failures,
        "warnings": warnings,
        "manifest_rows": len(manifest),
        "hpv_workbook_sheets": hpv_sheets,
        "gbd_files": [p.name for p in gbd_files],
    }
    output = ROOT / "05_results" / "validation_status.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if failures else 0


def analyse(uncertainty_draws: int, leave_one_out: bool) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from estimation_report import run_analysis

    report = run_analysis(
        ROOT, uncertainty_draws=uncertainty_draws, leave_one_out=leave_one_out
    )
    decisions = report["decision_rules"]
    print(json.dumps({
        "stage": "analyse",
        "treated_units": report["sample"]["treated_units"],
        "treated_countries": report["sample"]["treated_countries"],
        "decision_rules_failed": decisions["n_failed"],
        "causal_conclusion_permitted": decisions["causal_conclusion_permitted"],
        "verdict": decisions["verdict"],
        "outputs": [
            "05_results/effect_estimates.json",
            "05_results/effect_estimation_report.md",
            "07_tables/headline_estimates.csv",
            "07_tables/event_study_estimates.csv",
            "06_figures/Figure_1_event_study.png",
        ],
    }, indent=2))
    if not decisions["causal_conclusion_permitted"]:
        print(
            "STOP: estimates were produced, but the protocol's prespecified "
            "decision rules forbid a causal conclusion. See "
            "05_results/effect_estimation_report.md.",
            file=sys.stderr,
        )
        return 4
    return 0


def validate_outcome() -> int:
    """Compare the modelled outcome against observed CI5 registry incidence."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from ci5_validation import run_ci5_validation

    report = run_ci5_validation(ROOT)
    summary = report["summary"]
    print(json.dumps({
        "stage": "ci5-validation",
        "comparable_country_age_cells": summary["comparable_country_age_cells"],
        "countries": summary["countries"],
        "programme_vs_other_gap": summary["programme_vs_other_gap"],
        "outputs": [
            "05_results/ci5_observed_vs_modelled.json",
            "05_results/ci5_observed_vs_modelled.md",
            "07_tables/ci5_observed_vs_modelled.csv",
        ],
    }, indent=2))
    return 0


def validate_cancer_over_time() -> int:
    """Run the in-time placebo on longitudinal observed registry incidence."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from cancer_over_time import run_cancer_over_time_validation

    report = run_cancer_over_time_validation(ROOT)
    summary = report["summary"]
    print(json.dumps({
        "stage": "validate-cancer-over-time",
        "registry_rows_1990_2023": summary["panel_rows_1990_2023"],
        "registry_populations": summary["registry_populations"],
        "countries": summary["iso3_countries"],
        "genuine_post_treatment_rows": summary["genuine_post_treatment_rows"],
        "verdict": report["verdict"],
        "outputs": [
            "03_processed_data/iarc_cancer_over_time_cervix_female.csv",
            "05_results/cancer_over_time_validation.json",
            "05_results/cancer_over_time_validation.md",
            "07_tables/cancer_over_time_placebo_estimates.csv",
        ],
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "validate", "build", "analyse", "validate-outcome",
            "validate-cancer-over-time", "all",
        ],
        default="validate",
    )
    parser.add_argument(
        "--uncertainty-draws", type=int, default=0,
        help="Monte Carlo draws propagating the modelled-outcome interval "
             "into the overall ATT. 0 skips it.",
    )
    parser.add_argument(
        "--no-leave-one-out", action="store_true",
        help="Skip the leave-one-country-out refits.",
    )
    args = parser.parse_args()
    if args.stage == "validate":
        return validate()
    if args.stage in {"build", "all"}:
        validation_status = validate()
        if validation_status:
            return validation_status
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from build_pipeline import run_build

        report = run_build(ROOT)
        print(json.dumps(report, indent=2))
        if args.stage == "all":
            return analyse(args.uncertainty_draws, not args.no_leave_one_out)
        return 0
    if args.stage == "validate-outcome":
        return validate_outcome()
    if args.stage == "validate-cancer-over-time":
        return validate_cancer_over_time()
    if args.stage == "analyse":
        return analyse(args.uncertainty_draws, not args.no_leave_one_out)
    raise AssertionError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
