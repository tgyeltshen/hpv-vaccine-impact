import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exposure import (  # noqa: E402
    InclusiveInterval,
    classify_eligibility,
    merged_union,
    outcome_cohort_interval,
    target_cohort_interval,
)


class ExposureAlgorithmTests(unittest.TestCase):
    def test_outcome_interval(self):
        self.assertEqual(outcome_cohort_interval(2020, 20, 24), InclusiveInterval(1996, 2000))

    def test_target_interval(self):
        self.assertEqual(target_cohort_interval(2010, 12, 13), InclusiveInterval(1997, 1998))

    def test_partial_overlap(self):
        result = classify_eligibility(
            InclusiveInterval(1996, 2000),
            [InclusiveInterval(1997, 1998)],
            national_scope=True,
            ambiguous=False,
        )
        self.assertEqual(result["eligible_full"], 0)
        self.assertEqual(result["eligible_partial"], 1)
        self.assertAlmostEqual(result["eligible_share"], 0.4)

    def test_full_overlap_after_union(self):
        intervals = [InclusiveInterval(1996, 1997), InclusiveInterval(1998, 2000)]
        self.assertEqual(merged_union(intervals), [InclusiveInterval(1996, 2000)])
        result = classify_eligibility(
            InclusiveInterval(1996, 2000), intervals,
            national_scope=True, ambiguous=False,
        )
        self.assertEqual(result["eligible_full"], 1)

    def test_subnational_cannot_be_primary_eligible(self):
        result = classify_eligibility(
            InclusiveInterval(1996, 2000),
            [InclusiveInterval(1996, 2000)],
            national_scope=False,
            ambiguous=False,
        )
        self.assertEqual(result["eligible_full"], 0)
        self.assertEqual(result["eligibility_ambiguous"], 1)

    def test_ambiguous_cannot_be_primary_eligible(self):
        result = classify_eligibility(
            InclusiveInterval(1996, 2000),
            [InclusiveInterval(1996, 2000)],
            national_scope=True,
            ambiguous=True,
        )
        self.assertEqual(result["eligible_full"], 0)
        self.assertEqual(result["eligibility_ambiguous"], 1)


if __name__ == "__main__":
    unittest.main()

