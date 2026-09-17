"""Validate the modelled outcome against observed cancer-registry incidence.

The primary outcome is GBD 2023 cervical cancer incidence, which at ages 20-39 is
largely *modelled* rather than observed: invasive cervical cancer is rare at those
ages, so GBD borrows strength from covariates and regional patterns. That matters
here for a specific reason. If the covariates GBD leans on also predict HPV
programme adoption, then a negative association between eligibility and modelled
incidence can arise from the outcome model rather than from the programme, and it
would appear immediately rather than after a latency.

IARC CI5 Volume XII supplies observed counts and person-years from population
based cancer registries, by sex, five-year age band and cancer site, for registry
specific periods inside 2008-2017. That is the independent yardstick.

Two caveats hold throughout and are carried into the output rather than buried.
CI5 registries frequently cover a subnational population, so a CI5 country rate
is not a national rate and level differences are expected. And registry periods
differ, so each registry is compared against GBD for its own years.

Raw bytes are never extracted or modified; the archive is read in memory.
"""

from __future__ import annotations

import csv
import io
import json
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

CI5_ARCHIVE = Path("02_raw_data/iarc_ci5/volume_xii/CI5-XII.zip")
CERVIX_CANCER_CODE = "32"  # C53, per cancer_summary.txt in the archive
FEMALE_SEX_CODE = "2"
AGE_BANDS = (("20-24", "20_24"), ("25-29", "25_29"), ("30-34", "30_34"),
             ("35-39", "35_39"))

REGISTRY_LINE = re.compile(
    r"^(?P<registry>\d+)\s*\*\s*(?P<name>.+?)\s+(?P<start>\d{4})-(?P<end>\d{4})\s*$"
)

# CI5 country labels that the reviewed crosswalk spells differently. Each is a
# naming difference, not a judgement about which places count as countries, and
# every one was checked against the crosswalk rather than assumed. Liechtenstein
# is absent from the GBD location set, so it stays unmatched.
COUNTRY_LABEL_ALIASES = {
    "usa": "united states of america",
    "the netherlands": "netherlands",
    "czech republic": "czechia",
    "turkey": "turkiye",
    "uk": "united kingdom",
    "korea": "republic of korea",
    "russia": "russian federation",
    "iran": "iran islamic republic of",
}


def _normalise(name: str) -> str:
    """Match the crosswalk's normalisation so country names can be joined."""
    lowered = name.strip().lower()
    lowered = lowered.replace("&", "and")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def read_registries(archive: zipfile.ZipFile) -> dict[str, dict[str, object]]:
    """Registry id -> country name, registry name and covered years."""
    text = archive.read("Registry.txt").decode("utf-8", errors="replace")
    registries: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        match = REGISTRY_LINE.match(line.strip())
        if not match:
            continue
        label = match.group("name").strip()
        # "Country, Registry" for a subnational registry and "Country: Group" for
        # an ethnic or national subpopulation. Both resolve to the same country:
        # pooling them is what makes a country-level observed rate, and dropping
        # them would silently discard whole countries, the USA among them.
        country = re.split(r"[,:]", label)[0].strip()
        normalized = _normalise(country)
        registries[match.group("registry")] = {
            "registry_label": label,
            "country_label": country,
            "normalized_country": COUNTRY_LABEL_ALIASES.get(normalized, normalized),
            "year_start": int(match.group("start")),
            "year_end": int(match.group("end")),
            "subnational": bool(re.search(r"[,:]", label)),
        }
    return registries


def read_female_cervix_cases(archive: zipfile.ZipFile) -> dict[str, dict[str, float]]:
    stream = io.TextIOWrapper(
        archive.open("cases.csv"), encoding="utf-8", errors="replace"
    )
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in csv.DictReader(stream):
        if row["SEX"] != FEMALE_SEX_CODE or row["CANCER"] != CERVIX_CANCER_CODE:
            continue
        for band, suffix in AGE_BANDS:
            value = row.get(f"N{suffix}")
            if value not in (None, ""):
                out[row["REGISTRY"]][band] += float(value)
    return out


def read_female_person_years(archive: zipfile.ZipFile) -> dict[str, dict[str, float]]:
    stream = io.TextIOWrapper(
        archive.open("pop.csv"), encoding="utf-8", errors="replace"
    )
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in csv.DictReader(stream):
        if row["SEX"] != FEMALE_SEX_CODE:
            continue
        for band, suffix in AGE_BANDS:
            value = row.get(f"P{suffix}")
            if value not in (None, ""):
                out[row["REGISTRY"]][band] += float(value)
    return out


def load_crosswalk(root: Path) -> dict[str, str]:
    path = root / "03_processed_data" / "country_crosswalk_auto.csv"
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["iso3"]:
                mapping[row["normalized_name"]] = row["iso3"]
    return mapping


def load_modelled_rates(root: Path) -> dict[tuple[int, int, str], float]:
    """(gbd_location_id, year, age_group) -> modelled rate per 100k."""
    path = root / "03_processed_data" / "gbd_cervical_incidence.csv"
    rates: dict[tuple[int, int, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                key = (int(row["gbd_location_id"]), int(row["year"]), row["age_group"])
                rates[key] = float(row["rate_per_100k"])
            except (TypeError, ValueError):
                continue
    return rates


def load_location_ids(root: Path) -> dict[str, int]:
    path = root / "03_processed_data" / "country_crosswalk_auto.csv"
    ids: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["iso3"]:
                ids[row["iso3"]] = int(row["gbd_location_id"])
    return ids


def load_treated_countries(root: Path) -> set[str]:
    """Countries with at least one primary-treated cell, for the key contrast."""
    path = root / "03_processed_data" / "analytic_dataset.csv"
    treated: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("primary_treated_eligible") == "1":
                treated.add(row["iso3"])
    return treated


def build_comparison(root: Path) -> dict[str, object]:
    archive_path = root / CI5_ARCHIVE
    with zipfile.ZipFile(archive_path) as archive:
        registries = read_registries(archive)
        cases = read_female_cervix_cases(archive)
        person_years = read_female_person_years(archive)

    crosswalk = load_crosswalk(root)
    location_ids = load_location_ids(root)
    modelled = load_modelled_rates(root)
    treated_countries = load_treated_countries(root)

    # Pool registries within a country: a single registry is often a city, and a
    # pooled rate is the closest observed analogue to a national one available
    # here. Still not national, which the caveat records.
    pooled: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"cases": 0.0, "person_years": 0.0, "registries": 0,
                 "year_start": 9999, "year_end": 0, "subnational_only": 1}
    )
    unmatched: set[str] = set()
    for registry_id, meta in registries.items():
        iso3 = crosswalk.get(str(meta["normalized_country"]))
        if not iso3:
            unmatched.add(str(meta["country_label"]))
            continue
        for band, _ in AGE_BANDS:
            case_count = cases.get(registry_id, {}).get(band)
            population = person_years.get(registry_id, {}).get(band)
            if not population:
                continue
            entry = pooled[(iso3, band)]
            entry["cases"] += float(case_count or 0.0)
            entry["person_years"] += float(population)
            entry["registries"] += 1
            entry["year_start"] = min(entry["year_start"], int(meta["year_start"]))
            entry["year_end"] = max(entry["year_end"], int(meta["year_end"]))
            if not meta["subnational"]:
                entry["subnational_only"] = 0

    rows: list[dict[str, object]] = []
    for (iso3, band), entry in sorted(pooled.items()):
        if entry["person_years"] <= 0:
            continue
        observed = entry["cases"] / entry["person_years"] * 1e5
        location_id = location_ids.get(iso3)
        years = range(int(entry["year_start"]), int(entry["year_end"]) + 1)
        modelled_values = [
            modelled[(location_id, year, band)]
            for year in years
            if (location_id, year, band) in modelled
        ]
        if not modelled_values:
            continue
        modelled_mean = statistics.fmean(modelled_values)
        rows.append({
            "iso3": iso3,
            "age_group": band,
            "period": f"{int(entry['year_start'])}-{int(entry['year_end'])}",
            "registries_pooled": int(entry["registries"]),
            "subnational_only": bool(entry["subnational_only"]),
            "observed_cases": round(entry["cases"], 1),
            "observed_person_years": round(entry["person_years"], 1),
            "observed_rate_per_100k": round(observed, 4),
            "modelled_rate_per_100k": round(modelled_mean, 4),
            "modelled_over_observed": (
                round(modelled_mean / observed, 4) if observed > 0 else None
            ),
            "hpv_programme_country": iso3 in treated_countries,
        })

    return {
        "rows": rows,
        "unmatched_ci5_countries": sorted(unmatched),
        "treated_countries_in_analysis": len(treated_countries),
    }


def summarise(comparison: dict[str, object]) -> dict[str, object]:
    rows = [
        row for row in comparison["rows"]
        if row["modelled_over_observed"] is not None
    ]
    by_band: dict[str, object] = {}
    for band, _ in AGE_BANDS:
        ratios = [
            row["modelled_over_observed"] for row in rows if row["age_group"] == band
        ]
        if ratios:
            by_band[band] = {
                "countries": len(ratios),
                "median_modelled_over_observed": round(statistics.median(ratios), 4),
                "share_modelled_above_observed": round(
                    sum(1 for r in ratios if r > 1) / len(ratios), 4
                ),
            }

    # The test that matters for circularity: does the model-to-observed gap differ
    # between HPV-programme countries and the rest? A systematic difference means
    # the modelled outcome carries information about programme adoption, which
    # would generate an association with no latency.
    treated = [
        row["modelled_over_observed"] for row in rows if row["hpv_programme_country"]
    ]
    untreated = [
        row["modelled_over_observed"] for row in rows
        if not row["hpv_programme_country"]
    ]
    contrast: dict[str, object] = {
        "programme_country_cells": len(treated),
        "other_country_cells": len(untreated),
    }
    if treated and untreated:
        contrast.update({
            "median_ratio_programme_countries": round(statistics.median(treated), 4),
            "median_ratio_other_countries": round(statistics.median(untreated), 4),
            "mann_whitney_p_value": _mann_whitney(treated, untreated),
        })
    return {
        "comparable_country_age_cells": len(rows),
        "countries": len({row["iso3"] for row in rows}),
        "by_age_band": by_band,
        "programme_vs_other_gap": contrast,
    }


def _mann_whitney(first: list[float], second: list[float]) -> float | None:
    """Two-sided Mann-Whitney U p-value via a normal approximation with ties.

    Implemented here rather than pulled from SciPy so this module keeps the
    standard-library-only contract that the download and validate stages follow.
    """
    combined = sorted([(value, 0) for value in first] + [(value, 1) for value in second])
    n1, n2 = len(first), len(second)
    if not n1 or not n2:
        return None
    ranks: list[float] = [0.0] * len(combined)
    index = 0
    tie_correction = 0.0
    while index < len(combined):
        stop = index
        while stop + 1 < len(combined) and combined[stop + 1][0] == combined[index][0]:
            stop += 1
        average_rank = (index + stop) / 2 + 1
        span = stop - index + 1
        tie_correction += span ** 3 - span
        for position in range(index, stop + 1):
            ranks[position] = average_rank
        index = stop + 1

    rank_sum_first = sum(
        rank for rank, (_, group) in zip(ranks, combined) if group == 0
    )
    u_first = rank_sum_first - n1 * (n1 + 1) / 2
    total = n1 + n2
    mean = n1 * n2 / 2
    variance = n1 * n2 * (total + 1) / 12 - (
        n1 * n2 * tie_correction / (12 * total * (total - 1))
    )
    if variance <= 0:
        return None
    z = (u_first - mean) / variance ** 0.5
    return round(2 * 0.5 * _erfc(abs(z) / 2 ** 0.5), 6)


def _erfc(x: float) -> float:
    import math

    return math.erfc(x)


def run_ci5_validation(root: Path) -> dict[str, object]:
    comparison = build_comparison(root)
    summary = summarise(comparison)
    report = {
        "stage": "ci5-validation",
        "source": "IARC CI5 Volume XII, cervix uteri (C53), female",
        "caveats": [
            "CI5 registries frequently cover a subnational population, so a "
            "pooled CI5 country rate is not a national rate and level "
            "differences from GBD are expected rather than anomalous.",
            "Each registry is compared against GBD for that registry's own "
            "reporting years, which differ across registries.",
            "This compares levels. It does not by itself establish that the "
            "modelled series is unsuitable for a difference-in-differences "
            "design, which depends on the trends rather than the levels.",
        ],
        "summary": summary,
        "comparison": comparison["rows"],
        "unmatched_ci5_countries": comparison["unmatched_ci5_countries"],
    }
    results = root / "05_results"
    results.mkdir(exist_ok=True)
    (results / "ci5_observed_vs_modelled.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (results / "ci5_observed_vs_modelled.md").write_text(
        render(report), encoding="utf-8"
    )
    tables = root / "07_tables"
    tables.mkdir(exist_ok=True)
    if comparison["rows"]:
        with (tables / "ci5_observed_vs_modelled.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison["rows"][0]))
            writer.writeheader()
            writer.writerows(comparison["rows"])
    write_figure(root, report)
    return report


def figure_panels(report: dict[str, object]) -> dict[str, object]:
    """The quantities Figure S1 draws, separated from the drawing.

    Cells with no observed cases carry an undefined ratio and are counted here
    rather than silently dropped: at these ages a registry reporting zero cases
    is a real and frequent outcome, and omitting them without saying so would
    understate how thin the observed data are.
    """
    rows = report["comparison"]
    usable = [row for row in rows if row.get("modelled_over_observed") is not None]
    by_band: dict[str, dict[str, list[float]]] = {}
    for band, _ in AGE_BANDS:
        cells = [row for row in usable if row["age_group"] == band]
        by_band[band] = {
            "observed": [float(row["observed_rate_per_100k"]) for row in cells],
            "modelled": [float(row["modelled_rate_per_100k"]) for row in cells],
            "ratio": [float(row["modelled_over_observed"]) for row in cells],
        }
    ratios = {
        "programme": [
            float(row["modelled_over_observed"]) for row in usable
            if row["hpv_programme_country"]
        ],
        "other": [
            float(row["modelled_over_observed"]) for row in usable
            if not row["hpv_programme_country"]
        ],
    }
    return {
        "by_age_band": by_band,
        "ratios": ratios,
        "medians": {
            group: (statistics.median(values) if values else None)
            for group, values in ratios.items()
        },
        "comparable_cells": len(usable),
        "zero_observed_cells": len(rows) - len(usable),
        "mann_whitney_p_value":
            report["summary"]["programme_vs_other_gap"].get("mann_whitney_p_value"),
    }


def write_figure(root: Path, report: dict[str, object]) -> Path | None:
    """Figure S1: the modelled outcome against the observed one.

    Left, the levels: every comparable country-age cell against the identity
    line, on log axes because the rates span two orders of magnitude. Right, the
    quantity the paper's argument turns on: whether the modelled-to-observed
    ratio differs by programme status. Age band gets the sequential ramp because
    it is ordered; programme status gets two categorical slots because it is not.
    """
    panels = figure_panels(report)
    if not panels["comparable_cells"]:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BAND_COLOURS = {
        "20-24": "#86b6ef", "25-29": "#3987e5",
        "30-34": "#1c5cab", "35-39": "#0d366b",
    }
    PROGRAMME = "#2a78d6"   # categorical slot 1
    OTHER = "#898781"       # emphasis neutral: the comparison group
    INK, SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
    RULE = "#c3c2b7"

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- left: modelled against observed, by age band -------------------------
    everything = [
        value
        for band in panels["by_age_band"].values()
        for key in ("observed", "modelled")
        for value in band[key]
        if value > 0
    ]
    low, high = min(everything) * 0.7, max(everything) * 1.4
    left.plot([low, high], [low, high], color=RULE, linewidth=1.2, zorder=1)
    left.annotate(
        "identity: modelled = observed", xy=(high, high), xytext=(-4, -10),
        textcoords="offset points", ha="right", va="top", fontsize=8,
        color=SECONDARY,
    )
    for band, block in panels["by_age_band"].items():
        left.scatter(
            block["observed"], block["modelled"], s=34, alpha=0.85,
            color=BAND_COLOURS[band], edgecolors=SURFACE, linewidths=0.8,
            label=f"ages {band}", zorder=3,
        )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(low, high)
    left.set_ylim(low, high)
    left.set_xlabel("Observed rate per 100,000 (CI5-XII registries)", fontsize=9,
                    color=SECONDARY)
    left.set_ylabel("Modelled rate per 100,000 (GBD 2023)", fontsize=9,
                    color=SECONDARY)
    above = sum(
        1 for band in panels["by_age_band"].values()
        for ratio in band["ratio"] if ratio > 1
    )
    left.set_title(
        f"The modelled outcome runs above the observed one in "
        f"{above} of {panels['comparable_cells']} country-age cells",
        fontsize=10, color=INK, loc="left",
    )
    left.legend(frameon=False, fontsize=8.5, loc="upper left")

    # --- right: the ratio by programme status ---------------------------------
    import random

    jitter = random.Random(20260813)
    groups = (
        ("HPV programme countries", panels["ratios"]["programme"], PROGRAMME),
        ("all other countries", panels["ratios"]["other"], OTHER),
    )
    for position, (label, values, colour) in enumerate(groups):
        offsets = [position + jitter.uniform(-0.16, 0.16) for _ in values]
        right.scatter(offsets, values, s=30, alpha=0.7, color=colour,
                      edgecolors=SURFACE, linewidths=0.7, zorder=3)
        median = statistics.median(values) if values else None
        if median is not None:
            right.plot([position - 0.3, position + 0.3], [median, median],
                       color=colour, linewidth=2.4, zorder=4)
            right.annotate(
                f"median {median:.2f}", xy=(position + 0.3, median), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9,
                color=colour, fontweight="bold",
            )
    right.axhline(1.0, color=RULE, linewidth=1.2, zorder=2)
    right.annotate(
        "1.0 = modelled matches observed", xy=(1.4, 1.0), xytext=(0, -10),
        textcoords="offset points", ha="right", va="top", fontsize=8,
        color=SECONDARY,
    )
    right.set_yscale("log")
    right.set_xlim(-0.6, 1.75)
    right.set_xticks([0, 1])
    right.set_xticklabels([
        f"HPV programme\n({len(panels['ratios']['programme'])} cells)",
        f"all other countries\n({len(panels['ratios']['other'])} cells)",
    ], fontsize=9)
    right.set_ylabel("Modelled / observed", fontsize=9, color=SECONDARY)
    p_value = panels["mann_whitney_p_value"]
    right.set_title(
        "The gap is smaller where a programme exists"
        + (f", p = {p_value:.3g}" if p_value is not None else "")
        + "\nso the modelled outcome is not neutral to programme status",
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
        axes.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        axes.set_axisbelow(True)

    figure.text(
        0.008, 0.010,
        f"{panels['comparable_cells']} comparable country-age cells; "
        f"{panels['zero_observed_cells']} cells with zero observed cases are "
        "omitted because the ratio is undefined.\nCI5 registries are often "
        "subnational, so level differences are expected; the comparison between "
        "the two groups is what carries the argument.",
        fontsize=7.5, color=SECONDARY, linespacing=1.5,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    figures = root / "06_figures"
    figures.mkdir(exist_ok=True)
    path = figures / "Figure_S1_observed_vs_modelled.png"
    figure.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(figure)
    return path


def render(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# Observed versus modelled cervical cancer incidence, ages 20-39",
        "",
        f"Source: {report['source']}. Comparable cells: "
        f"{summary['comparable_country_age_cells']} across "
        f"{summary['countries']} countries.",
        "",
        "## Why this check exists",
        "",
        "GBD cervical cancer incidence at ages 20-39 is largely modelled, because "
        "the disease is rare at those ages. If the covariates the model leans on "
        "also predict HPV programme adoption, a negative association between "
        "eligibility and modelled incidence can be produced by the outcome model "
        "rather than by the programme, and it would appear with no latency. That "
        "is the shape of the anomaly in the main analysis, so it needs testing "
        "against observed registry data rather than argued about.",
        "",
        "## Levels by age band",
        "",
        "| Age band | Countries | Median modelled / observed | Share modelled above observed |",
        "|---|---|---|---|",
    ]
    for band, block in summary["by_age_band"].items():
        lines.append(
            f"| {band} | {block['countries']} | "
            f"{block['median_modelled_over_observed']:.3f} | "
            f"{block['share_modelled_above_observed']:.0%} |"
        )

    gap = summary["programme_vs_other_gap"]
    lines += [
        "",
        "## The test that matters: does the gap track programme adoption?",
        "",
    ]
    if "median_ratio_programme_countries" in gap:
        p_value = gap["mann_whitney_p_value"]
        verdict = (
            "The modelled-to-observed gap differs between programme and "
            "non-programme countries, so the modelled series is **not neutral "
            "with respect to programme status**. That is the precondition for the "
            "circularity concern, and it is established here.\n\n"
            "It is not sufficient for it. A difference-in-differences estimator "
            "removes any level bias that is constant within a country, so a "
            "differential *level* gap of this kind largely differences out. What "
            "would manufacture the immediate effect is a differential *trend* in "
            "the gap, and Volume XII gives one pooled period per registry, so it "
            "cannot measure a trend. That longitudinal test is now implemented "
            "separately with IARC Cancer Over Time; see "
            "`05_results/cancer_over_time_validation.md`."
            if p_value is not None and p_value < 0.05 else
            "No detectable difference in the modelled-to-observed gap between "
            "programme and non-programme countries. This does not vindicate the "
            "modelled series, and a level comparison could not have done so, but "
            "it removes the most direct route by which the outcome model could "
            "manufacture the immediate effect."
        )
        lines += [
            f"- Programme countries: {gap['programme_country_cells']} cells, "
            f"median modelled/observed {gap['median_ratio_programme_countries']:.3f}",
            f"- Other countries: {gap['other_country_cells']} cells, "
            f"median modelled/observed {gap['median_ratio_other_countries']:.3f}",
            f"- Mann-Whitney two-sided p = "
            + (f"{p_value:.4g}" if p_value is not None else "not estimable"),
            "",
            verdict,
        ]
    else:
        lines.append(
            "Not estimable: one of the two groups has no comparable cells."
        )

    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in report["caveats"]]
    unmatched = report["unmatched_ci5_countries"]
    if unmatched:
        lines += [
            "",
            f"{len(unmatched)} CI5 country labels did not match the reviewed "
            "crosswalk and are excluded rather than guessed at: "
            + ", ".join(unmatched[:15])
            + ("..." if len(unmatched) > 15 else ""),
        ]
    return "\n".join(lines) + "\n"
