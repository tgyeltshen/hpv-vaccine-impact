"""Tests for the two supplementary figures' underlying quantities.

A figure is an argument, so what is tested here is the argument: which cells
enter it, which are excluded and why, and whether the summaries drawn on it are
the summaries the report states. The drawing itself is covered only by a smoke
test that the file is written, because pixel comparisons fail on font changes
and would say nothing about whether the figure is right.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ci5_validation as C  # noqa: E402
import estimation_report as E  # noqa: E402


def ci5_report() -> dict[str, object]:
    """Four cells with a known ratio each, plus one with no observed cases."""
    rows = [
        {"iso3": "AAA", "age_group": "20-24", "observed_rate_per_100k": 2.0,
         "modelled_rate_per_100k": 4.0, "modelled_over_observed": 2.0,
         "hpv_programme_country": True},
        {"iso3": "BBB", "age_group": "20-24", "observed_rate_per_100k": 4.0,
         "modelled_rate_per_100k": 4.0, "modelled_over_observed": 1.0,
         "hpv_programme_country": True},
        {"iso3": "CCC", "age_group": "35-39", "observed_rate_per_100k": 10.0,
         "modelled_rate_per_100k": 30.0, "modelled_over_observed": 3.0,
         "hpv_programme_country": False},
        {"iso3": "DDD", "age_group": "35-39", "observed_rate_per_100k": 10.0,
         "modelled_rate_per_100k": 50.0, "modelled_over_observed": 5.0,
         "hpv_programme_country": False},
        {"iso3": "EEE", "age_group": "25-29", "observed_rate_per_100k": 0.0,
         "modelled_rate_per_100k": 1.5, "modelled_over_observed": None,
         "hpv_programme_country": False},
    ]
    return {
        "comparison": rows,
        "summary": {"programme_vs_other_gap": {"mann_whitney_p_value": 0.04}},
    }


def test_cells_with_no_observed_cases_are_excluded_and_counted():
    """A registry reporting zero cases has no ratio; dropping it silently would
    make the observed data look better attested than it is."""
    panels = C.figure_panels(ci5_report())
    assert panels["comparable_cells"] == 4
    assert panels["zero_observed_cells"] == 1
    assert all(
        ratio is not None
        for band in panels["by_age_band"].values()
        for ratio in band["ratio"]
    )
    # The excluded cell's age band is present but empty, not missing.
    assert panels["by_age_band"]["25-29"]["ratio"] == []


def test_group_medians_are_the_medians_of_the_plotted_points():
    panels = C.figure_panels(ci5_report())
    assert panels["ratios"]["programme"] == [2.0, 1.0]
    assert panels["ratios"]["other"] == [3.0, 5.0]
    for group, values in panels["ratios"].items():
        assert panels["medians"][group] == pytest.approx(statistics.median(values))
    assert panels["medians"]["programme"] < panels["medians"]["other"]


def test_scatter_coordinates_reproduce_each_cell_ratio():
    """The left panel and the right panel have to be showing the same numbers."""
    panels = C.figure_panels(ci5_report())
    for block in panels["by_age_band"].values():
        for observed, modelled, ratio in zip(
            block["observed"], block["modelled"], block["ratio"]
        ):
            assert modelled / observed == pytest.approx(ratio)


def test_ci5_figure_is_written(tmp_path):
    path = C.write_figure(tmp_path, ci5_report())
    assert path is not None and path.exists()
    assert path.stat().st_size > 10_000


def test_no_figure_when_nothing_is_comparable(tmp_path):
    empty = {
        "comparison": [],
        "summary": {"programme_vs_other_gap": {}},
    }
    assert C.write_figure(tmp_path, empty) is None


def event_study(estimate: float) -> list[dict[str, object]]:
    rows = [
        {"event_time": time, "estimate": (point := estimate if time >= 0 else 0.0),
         "std_error": 0.01, "conf_low": point - 0.02, "conf_high": point + 0.02}
        for time in range(-10, 7)
    ]
    rows.append({"event_time": 7, "estimate": None, "std_error": None})
    return rows


def placebo_report() -> dict[str, object]:
    def block(estimate: float) -> dict[str, object]:
        return {
            "event_study": event_study(estimate),
            "overall_post_att": estimate,
            "overall_post_conf_low": estimate - 0.03,
            "overall_post_conf_high": estimate + 0.03,
        }

    return {
        "sample": {"treated_units": 29},
        "specifications": {"sun_abraham_log_rate": block(-0.03)},
        "falsification_tests": {
            "in_time_lead_8": {"treated_units": 29, "estimate": block(-0.10)},
            "in_time_lead_12": {"treated_units": 29, "estimate": block(-0.08)},
            "in_time_lead_8_prophylactic_age_15": {
                "treated_units": 22, "estimate": block(-0.09)
            },
        },
    }


def test_placebo_series_marks_which_runs_contain_no_treatment():
    series = E.placebo_series(placebo_report())["series"]
    assert [row["is_placebo"] for row in series] == [False, True, True, True]
    assert [row["key"] for row in series][0] == "sun_abraham_log_rate"
    # Treated-unit counts come from the run itself, not from the primary sample.
    assert [row["treated_units"] for row in series] == [29, 29, 29, 22]


def test_unestimated_event_times_are_dropped_before_plotting():
    series = E.placebo_series(placebo_report())["series"]
    for row in series:
        assert all(point["estimate"] is not None for point in row["event_study"])
        assert 7 not in [point["event_time"] for point in row["event_study"]]


def test_placebo_series_skips_runs_the_pipeline_did_not_produce():
    report = placebo_report()
    del report["falsification_tests"]["in_time_lead_12"]
    keys = [row["key"] for row in E.placebo_series(report)["series"]]
    assert "in_time_lead_12" not in keys
    assert len(keys) == 3


def test_placebo_figure_is_written(tmp_path):
    path = E.write_placebo_figure(tmp_path, placebo_report())
    assert path is not None and path.exists()
    assert path.stat().st_size > 10_000


def test_no_placebo_figure_without_a_placebo_run(tmp_path):
    report = placebo_report()
    report["falsification_tests"] = {}
    assert E.write_placebo_figure(tmp_path, report) is None
