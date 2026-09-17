import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import maturity as M  # noqa: E402
from exposure import outcome_cohort_interval  # noqa: E402


def test_target_age_upper_parses_the_range_form():
    assert M.target_age_upper("9-13") == 13
    assert M.target_age_upper("12") == 12
    assert M.target_age_upper(" 11-14 ") == 14
    assert M.target_age_upper("") is None
    assert M.target_age_upper(None) is None
    assert M.target_age_upper("nan") is None


def test_maturity_year_agrees_with_the_frozen_cohort_primitives():
    """The returned year must be the first one the exposure algorithm would accept.

    This is the test that matters: the forecast restates cohort arithmetic that
    already exists in ``exposure``, so it has to be pinned to that definition
    rather than to its own algebra.
    """
    introduction, target = 2008, 12
    for lower, upper in M.OUTCOME_BANDS:
        year = M.maturity_year(introduction, target, upper)
        oldest = M.oldest_eligible_cohort(introduction, target)

        matured = outcome_cohort_interval(year, lower, upper)
        assert matured.lower >= oldest

        # and it is the *first* such year
        year_before = outcome_cohort_interval(year - 1, lower, upper)
        assert year_before.lower < oldest


def test_adult_catch_up_is_capped_so_a_band_cannot_mature_before_the_programme():
    """Uncapped, a catch-up to age 26 matures the 20-24 band before introduction."""
    uncapped = M.maturity_year(2010, 26, 24, prophylactic_age_cap=99)
    assert uncapped < 2010

    capped = M.maturity_year(2010, 26, 24)
    assert capped > 2010
    assert capped == 2010 - M.PROPHYLACTIC_AGE_CAP + 24


def test_later_bands_mature_later_and_earlier_programmes_mature_sooner():
    years = [M.maturity_year(2008, 12, upper) for _, upper in M.OUTCOME_BANDS]
    assert years == sorted(years)
    assert len(set(years)) == len(years)

    assert M.maturity_year(2006, 12, 34) < M.maturity_year(2015, 12, 34)


def test_forecast_skips_countries_without_a_documented_programme():
    countries = pd.DataFrame([
        {"iso3": "AAA", "first_national_introduction_year": 2008,
         "current_target_age": "9-13"},
        {"iso3": "BBB", "first_national_introduction_year": None,
         "current_target_age": "12"},
        {"iso3": "CCC", "first_national_introduction_year": 2010,
         "current_target_age": ""},
    ])
    forecast = M.cohort_maturity_forecast(countries)

    assert set(forecast["iso3"]) == {"AAA"}
    assert len(forecast) == len(M.OUTCOME_BANDS)


def test_summary_counts_are_monotone_in_the_milestone_year():
    countries = pd.DataFrame([
        {"iso3": f"C{index:02d}", "first_national_introduction_year": year,
         "current_target_age": "12"}
        for index, year in enumerate(range(2006, 2020))
    ])
    summary = M.maturity_summary(M.cohort_maturity_forecast(countries))

    for band in summary["bands"].values():
        counts = list(band["countries_matured_by"].values())
        assert counts == sorted(counts)
        assert counts[-1] <= summary["countries"]

    # A later band cannot have matured in more countries than an earlier one.
    by_2040 = {
        name: block["countries_matured_by"]["2040"]
        for name, block in summary["bands"].items()
    }
    assert by_2040["20-24"] >= by_2040["25-29"] >= by_2040["30-34"] >= by_2040["35-39"]


def test_forecast_on_the_real_country_table_leaves_no_band_mature_at_30_34():
    """Guards the paper's headline claim against a silent data change."""
    root = Path(__file__).resolve().parents[2]
    countries = M.load_country_table(root)
    forecast = M.cohort_maturity_forecast(countries)

    assert forecast["iso3"].nunique() > 100
    band = forecast.loc[forecast["age_group"] == "30-34", "maturity_year"]
    assert (band <= 2023).sum() == 0
    assert band.min() == 2026
