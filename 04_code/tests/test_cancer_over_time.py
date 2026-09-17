import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cancer_over_time import build_registry_panel, validate_payloads  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def test_registered_payloads_have_the_requested_site_sex_ages_and_type():
    populations, cancers, dataset = validate_payloads(ROOT)
    assert len(populations) == 84
    assert next(row for row in cancers if row["id"] == 16)["ICD"] == "C53"
    assert {row["age_label"] for row in dataset} == {
        "20-24", "25-29", "30-34", "35-39"
    }
    assert {row["sex"] for row in dataset} == {2}
    assert {row["type"] for row in dataset} == {0}
    assert {row["cancer"] for row in dataset} == {16}


def test_registry_panel_is_unique_and_preserves_zero_rates():
    panel, summary = build_registry_panel(ROOT)
    assert len(panel) == 5036
    assert summary["registry_populations"] == 52
    assert summary["iso3_countries"] == 47
    assert summary["zero_rate_rows"] == 537
    assert not panel.duplicated(
        ["registry_population_id", "age_group", "year"]
    ).any()
    assert np.isfinite(panel["asinh_observed_rate"]).all()


def test_treatment_linkage_is_country_age_specific_and_not_registry_inferred():
    panel, summary = build_registry_panel(ROOT)
    analytic = pd.read_csv(ROOT / "03_processed_data/analytic_dataset.csv")
    expected = (
        analytic[["iso3", "age_group", "treatment_cohort_year"]]
        .drop_duplicates()
        .set_index(["iso3", "age_group"])["treatment_cohort_year"]
        .to_dict()
    )
    for _, row in panel.iterrows():
        assert row["treatment_cohort_year"] == expected.get(
            (row["iso3"], row["age_group"]), 0
        )
    assert summary["treated_iso3_countries"] == 15
    assert summary["genuine_post_treatment_rows"] == 4


def test_payload_validator_rejects_wrong_cancer(tmp_path, monkeypatch):
    source = ROOT / "02_raw_data/iarc_cancer_over_time/2026-08-13_api"
    target = tmp_path / "02_raw_data/iarc_cancer_over_time/2026-08-13_api"
    target.mkdir(parents=True)
    for name in ("meta_populations.json", "meta_cancers.json"):
        (target / name).write_bytes((source / name).read_bytes())
    document = json.loads((source / "cervix_female_ages20_39_annual.json").read_text())
    document["dataset"][0]["cancer"] = 99
    (target / "cervix_female_ages20_39_annual.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unexpected site"):
        validate_payloads(tmp_path)
