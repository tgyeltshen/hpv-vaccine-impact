import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_pipeline import resolve_jrf_target_ages  # noqa: E402


def schedule_row(**overrides):
    row = {
        "iso3": "AAA", "who_region": "EURO", "year": 2012, "vaccine_code": "HPV4",
        "schedule_round": 1, "target_population_code": "FEMALE",
        "geographic_scope": "NATIONAL", "age_administered_native": "Y12",
        "target_age_lower": 12, "target_age_upper": 12,
        "source_comment": "", "source_row": 2,
    }
    row.update(overrides)
    return row


class JrfTargetAgeResolutionTests(unittest.TestCase):
    intro = {"AAA": 2010}
    coverage = {"AAA": {2011, 2012}}

    def resolve(self, rows, intro=None, coverage=None):
        return resolve_jrf_target_ages(
            rows,
            self.intro if intro is None else intro,
            self.coverage if coverage is None else coverage,
        )

    def test_labelled_routine_target_is_accepted(self):
        resolved, _ = self.resolve([schedule_row()])
        self.assertEqual(resolved[("AAA", 2012)]["age"], (12, 12))
        self.assertEqual(
            resolved[("AAA", 2012)]["provenance"], "jrf_labelled_target_population"
        )

    def test_later_rounds_are_not_target_ages(self):
        """Round 2+ rows carry dose intervals such as +M6, never target ages."""
        rows = [schedule_row(schedule_round=2, age_administered_native="+M6",
                             target_age_lower="", target_age_upper="")]
        resolved, _ = self.resolve(rows)
        self.assertEqual(resolved, {})

    def test_subnational_scope_is_excluded(self):
        resolved, _ = self.resolve([schedule_row(geographic_scope="SUBNATIONAL")])
        self.assertEqual(resolved, {})

    def test_non_routine_target_populations_are_excluded(self):
        for code in ("PLANNED", "RISKGROUPS", "ADULTS"):
            with self.subTest(code=code):
                resolved, _ = self.resolve([schedule_row(target_population_code=code)])
                self.assertEqual(resolved, {})

    def test_rows_after_the_outcome_panel_are_excluded(self):
        resolved, _ = self.resolve([schedule_row(year=2025)])
        self.assertEqual(resolved, {})

    def test_blank_target_population_needs_corroboration(self):
        """A blank TARGETPOP is never imputed; without a national introduction
        and evidence of a real female programme the country-year is dropped."""
        rows = [schedule_row(target_population_code="")]
        resolved, _ = self.resolve(rows, intro={}, coverage={})
        self.assertEqual(resolved, {})

    def test_blank_target_population_before_introduction_is_excluded(self):
        rows = [schedule_row(year=2008, target_population_code="")]
        resolved, _ = self.resolve(rows)
        self.assertEqual(resolved, {})

    def test_blank_target_population_corroborated_same_year(self):
        rows = [schedule_row(target_population_code="")]
        resolved, _ = self.resolve(rows)
        self.assertEqual(
            resolved[("AAA", 2012)]["provenance"],
            "jrf_unlabelled_corroborated_same_year",
        )

    def test_blank_target_population_before_wuenic_coverage_starts(self):
        """WUENIC begins in 2010, so 2006-2009 programme years can only be
        corroborated at programme level. They must not be silently dropped."""
        rows = [schedule_row(year=2010, target_population_code="")]
        resolved, _ = self.resolve(rows)
        self.assertEqual(
            resolved[("AAA", 2010)]["provenance"],
            "jrf_unlabelled_corroborated_programme_level",
        )

    def test_products_in_one_country_year_union_their_ages(self):
        rows = [
            schedule_row(vaccine_code="HPV4", target_age_lower=12, target_age_upper=12),
            schedule_row(vaccine_code="HPV9", target_age_lower=9, target_age_upper=14),
        ]
        resolved, stats = self.resolve(rows)
        self.assertEqual(resolved[("AAA", 2012)]["age"], (9, 14))
        self.assertTrue(resolved[("AAA", 2012)]["products_disagreed"])
        self.assertEqual(stats["country_years_with_product_age_disagreement"], 1)

    def test_catchup_is_flagged(self):
        resolved, _ = self.resolve([schedule_row(target_population_code="CATCHUP_C")])
        self.assertTrue(resolved[("AAA", 2012)]["catchup"])


if __name__ == "__main__":
    unittest.main()
