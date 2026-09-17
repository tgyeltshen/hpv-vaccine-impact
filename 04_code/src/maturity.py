"""When vaccinated birth cohorts reach the ages at which cervical cancer is measurable.

The estimation stage reports that the treated panel is immature. That is a
qualitative statement until someone asks *how* immature, and for how long. This
module turns it into dated arithmetic: for each country with a documented national
programme, the calendar year in which each outcome age band becomes wholly composed
of birth cohorts that were eligible at a prophylactically plausible age.

The arithmetic reuses the frozen cohort primitives in ``exposure`` rather than
restating them, so that a change to the exposure algorithm cannot silently leave
this forecast describing a different design.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exposure import outcome_cohort_interval

# Deviation D4 established that cohorts first reached in their twenties cannot have
# their invasive cervical cancer risk altered the way a cohort reached at 12 can.
# The forecast therefore answers "when do the bands fill with cohorts vaccinated
# young enough to matter", not "when do they fill with anyone a programme touched".
PROPHYLACTIC_AGE_CAP = 15

OUTCOME_BANDS = ((20, 24), (25, 29), (30, 34), (35, 39))


def target_age_upper(value: object) -> float | None:
    """Upper routine target age, tolerating the '9-13' range form in the source."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text.split("-")[-1])
    except ValueError:
        return None


def maturity_year(
    introduction_year: int,
    target_age: float,
    band_upper_age: int,
    *,
    prophylactic_age_cap: int = PROPHYLACTIC_AGE_CAP,
) -> int:
    """First calendar year in which an age band holds only eligible birth cohorts.

    A routine programme running from ``introduction_year`` at ages up to
    ``target_age`` leaves ``introduction_year - target_age`` as its oldest eligible
    birth cohort. An age band in calendar year ``y`` spans cohorts
    ``[y - band_upper_age, y - band_lower_age]``, so the band is wholly eligible
    once its oldest cohort is no older than that: ``y >= introduction_year -
    target_age + band_upper_age``.

    Catch-up campaigns reaching adults are capped at ``prophylactic_age_cap``. Left
    uncapped, a broad catch-up makes a band "mature" before the programme started,
    which is arithmetically true and biologically meaningless.
    """
    effective = min(float(target_age), float(prophylactic_age_cap))
    return int(round(introduction_year - effective + band_upper_age))


def oldest_eligible_cohort(
    introduction_year: int,
    target_age: float,
    *,
    prophylactic_age_cap: int = PROPHYLACTIC_AGE_CAP,
) -> int:
    effective = min(float(target_age), float(prophylactic_age_cap))
    return int(round(introduction_year - effective))


def cohort_maturity_forecast(
    countries: pd.DataFrame,
    *,
    prophylactic_age_cap: int = PROPHYLACTIC_AGE_CAP,
) -> pd.DataFrame:
    """One row per country x outcome band, with the year that band matures."""
    records = []
    for _, row in countries.iterrows():
        introduction = row.get("first_national_introduction_year")
        target = target_age_upper(row.get("current_target_age"))
        if pd.isna(introduction) or target is None:
            continue
        for lower, upper in OUTCOME_BANDS:
            records.append({
                "iso3": row["iso3"],
                "introduction_year": int(introduction),
                "target_age_upper": target,
                "age_group": f"{lower}-{upper}",
                "band_upper_age": upper,
                "maturity_year": maturity_year(
                    int(introduction), target, upper,
                    prophylactic_age_cap=prophylactic_age_cap,
                ),
            })
    return pd.DataFrame.from_records(records)


def maturity_summary(forecast: pd.DataFrame, milestones=(2023, 2030, 2040, 2050)) -> dict:
    """Countries whose band has matured by each milestone year."""
    summary = {"countries": int(forecast["iso3"].nunique()), "bands": {}}
    for age_group, block in forecast.groupby("age_group"):
        years = block["maturity_year"]
        summary["bands"][age_group] = {
            "earliest": int(years.min()),
            "median": float(years.median()),
            "countries_matured_by": {
                str(year): int((years <= year).sum()) for year in milestones
            },
        }
    return summary


def load_country_table(root: Path) -> pd.DataFrame:
    path = root / "03_processed_data" / "cohort_coverage_feasibility_by_country.csv"
    return pd.read_csv(path)
