"""Small tests for the data organization step.

Run from the project root:
    python data_organized/test_data/run_tests.py
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DATA_DIR.parent
sys.path.insert(0, str(DATA_DIR))

from prepocess_data import gap_to_winner_seconds


class RaceGapTests(unittest.TestCase):
    def test_winner_gap_is_zero(self):
        self.assertEqual(gap_to_winner_seconds(1, pd.Timedelta("1:32:05")), 0.0)

    def test_non_winner_gap_uses_reported_delta(self):
        self.assertEqual(gap_to_winner_seconds(2, pd.Timedelta(seconds=4.321)), 4.321)

    def test_missing_gap_stays_missing(self):
        self.assertTrue(np.isnan(gap_to_winner_seconds(12, pd.NaT)))


class OrganizedFileTests(unittest.TestCase):
    def test_core_csvs_exist(self):
        expected = [
            "qualifying_deltas.csv",
            "race_results.csv",
            "lap_time_spreads.csv",
            "driver_season_profiles.csv",
        ]
        for filename in expected:
            with self.subTest(filename=filename):
                self.assertTrue((DATA_DIR / filename).exists())

    def test_race_results_has_expected_columns(self):
        race_results = pd.read_csv(DATA_DIR / "race_results.csv", nrows=5)
        expected = {"Year", "Round", "Driver", "Team", "Position", "Points", "GapToWinner_s"}
        self.assertTrue(expected.issubset(race_results.columns))


if __name__ == "__main__":
    unittest.main(verbosity=2)
