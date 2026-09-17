import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import (  # noqa: E402
    normalize_country_name,
    parse_age_band,
    parse_hpv_target_age,
)


class IngestUtilityTests(unittest.TestCase):
    def test_gbd_age_band(self):
        self.assertEqual(parse_age_band("20-24 years"), (20, 24))
        self.assertIsNone(parse_age_band("All ages"))

    def test_hpv_target_age(self):
        self.assertEqual(parse_hpv_target_age("Y9-Y14"), (9, 14))
        self.assertEqual(parse_hpv_target_age("12"), (12, 12))
        self.assertIsNone(parse_hpv_target_age("+M6"))

    def test_country_normalization(self):
        self.assertEqual(normalize_country_name("Côte d’Ivoire"), "cote d ivoire")


if __name__ == "__main__":
    unittest.main()
