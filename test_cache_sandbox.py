"""
Sandbox tests for cache_fastf1_data.py

All FastF1 API calls are mocked - no network requests are made and
discovery_raw_data/ is never touched.  Run with:
    python test_cache_sandbox.py
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_mock_session(name="Race"):
    s = MagicMock()
    s.name = name
    return s


# ---------------------------------------------------------------------------
# Tests for fetch_session
# ---------------------------------------------------------------------------

class TestFetchSession(unittest.TestCase):

    def _run(self, side_effects):
        """
        Patch fastf1.get_session so each successive call to session.load()
        raises (or returns normally) according to side_effects.

        side_effects items:
          None                       - load() succeeds (returns None like real fastf1)
          an exception instance      - load() raises that exception
        """
        import cache_fastf1_data as m

        mock_session = _make_mock_session()
        load_side = []
        for effect in side_effects:
            if effect is None:
                load_side.append(None)  # normal return
            else:
                load_side.append(effect)

        mock_session.load.side_effect = load_side

        with patch.object(m.fastf1, "get_session", return_value=mock_session):
            # Patch time.sleep to skip actual waiting
            with patch("time.sleep"):
                return m.fetch_session(2024, 1, 1)

    def test_ok_on_first_try(self):
        import cache_fastf1_data as m
        status, session = self._run([None])
        self.assertEqual(status, "ok")
        self.assertIsNotNone(session)

    def test_not_found(self):
        import cache_fastf1_data as m
        mock_session = _make_mock_session()
        with patch.object(m.fastf1, "get_session", side_effect=ValueError("no such session")):
            with patch("time.sleep"):
                status, session = m.fetch_session(2024, 1, 6)
        self.assertEqual(status, "not_found")
        self.assertIsNone(session)

    def test_transient_error_then_ok(self):
        # Two transient failures, then success
        status, session = self._run([
            RuntimeError("network timeout"),
            RuntimeError("network timeout"),
            None,
        ])
        self.assertEqual(status, "ok")

    def test_transient_error_exhausted(self):
        # Fails on every transient attempt - should give up
        import cache_fastf1_data as m
        errors = [RuntimeError("bad")] * (len(m.RETRY_DELAYS) + 1)
        status, _ = self._run(errors)
        self.assertEqual(status, "failed")

    def test_rate_limit_then_ok(self):
        import cache_fastf1_data as m
        # One rate-limit hit, then success
        status, session = self._run([
            m.fastf1.RateLimitExceededError("ergast.com: 200 calls/h"),
            None,
        ])
        self.assertEqual(status, "ok")
        self.assertIsNotNone(session)

    def test_rate_limit_exhausted(self):
        import cache_fastf1_data as m
        # Keeps rate-limiting through the whole backoff schedule
        errors = [
            m.fastf1.RateLimitExceededError("ergast.com: 200 calls/h")
        ] * (len(m.RATE_LIMIT_BACKOFFS) + 1)
        status, _ = self._run(errors)
        self.assertEqual(status, "failed")

    def test_rate_limit_does_not_consume_transient_budget(self):
        """A rate-limit retry must not count against the transient retry counter."""
        import cache_fastf1_data as m
        # One rate-limit, then RETRY_DELAYS-many transient errors, then success
        errors = (
            [m.fastf1.RateLimitExceededError("ergast.com: 200 calls/h")]
            + [RuntimeError("timeout")] * len(m.RETRY_DELAYS)
            + [None]
        )
        status, session = self._run(errors)
        self.assertEqual(status, "ok")


# ---------------------------------------------------------------------------
# Tests for process_round and done_set state tracking
# ---------------------------------------------------------------------------

class TestProcessRound(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_state(self, m):
        """Return a context manager that redirects STATE_FILE to a temp path."""
        return patch.object(
            m, "STATE_FILE",
            os.path.join(self.tmpdir, "progress.json"),
        )

    def _make_fetch_results(self, results):
        """
        results: list of 'ok'|'not_found'|'failed' per session index 1..N.
        Returns a side_effect list for fetch_session.
        """
        out = []
        for r in results:
            if r == "ok":
                out.append(("ok", _make_mock_session()))
            elif r == "not_found":
                out.append(("not_found", None))
            else:
                out.append(("failed", None))
        return out

    def test_all_5_sessions_ok(self):
        import cache_fastf1_data as m
        side = self._make_fetch_results(["ok"] * 5)
        with self._patch_state(m):
            with patch.object(m, "fetch_session", side_effect=side):
                done = set()
                m.process_round(2024, 1, "Bahrain", done)
        # All 5 session keys must be present
        for i in range(1, 6):
            self.assertIn(m.session_key(2024, 1, i), done)

    def test_not_found_at_index_4_marks_4_and_5(self):
        """When session 4 is not_found, indices 4 AND 5 must be added to done_set."""
        import cache_fastf1_data as m
        side = self._make_fetch_results(["ok", "ok", "ok", "not_found"])
        with self._patch_state(m):
            with patch.object(m, "fetch_session", side_effect=side):
                done = set()
                m.process_round(2024, 1, "Bahrain", done)
        # ok sessions
        for i in range(1, 4):
            self.assertIn(m.session_key(2024, 1, i), done, f"session {i} should be done")
        # not_found - both 4 and 5 must be marked
        self.assertIn(m.session_key(2024, 1, 4), done, "session 4 not marked")
        self.assertIn(m.session_key(2024, 1, 5), done, "session 5 not marked")

    def test_not_found_at_index_1_marks_all(self):
        """not_found on the first session must mark all 5 indices."""
        import cache_fastf1_data as m
        side = self._make_fetch_results(["not_found"])
        with self._patch_state(m):
            with patch.object(m, "fetch_session", side_effect=side):
                done = set()
                m.process_round(2024, 99, "FutureRace", done)
        for i in range(1, 6):
            self.assertIn(m.session_key(2024, 99, i), done, f"index {i} should be marked")

    def test_already_done_sessions_are_skipped(self):
        """Sessions in done_set must not call fetch_session again."""
        import cache_fastf1_data as m
        # Pre-populate sessions 1-4 as done
        pre = {m.session_key(2024, 1, i) for i in range(1, 5)}
        # Only session 5 should be fetched
        side = self._make_fetch_results(["ok"])
        with self._patch_state(m):
            with patch.object(m, "fetch_session", side_effect=side) as mock_fetch:
                done = set(pre)
                m.process_round(2024, 1, "Bahrain", done)
        # fetch_session must have been called exactly once (for session 5)
        self.assertEqual(mock_fetch.call_count, 1)
        self.assertIn(m.session_key(2024, 1, 5), done)

    def test_failed_session_does_not_block_later_sessions(self):
        """A 'failed' session must be skipped; later sessions must still be tried."""
        import cache_fastf1_data as m
        side = self._make_fetch_results(["ok", "failed", "ok", "ok", "ok"])
        with self._patch_state(m):
            with patch.object(m, "fetch_session", side_effect=side):
                done = set()
                m.process_round(2024, 1, "Bahrain", done)
        # sessions 1, 3, 4, 5 should be done; session 2 failed so it's absent
        self.assertIn(m.session_key(2024, 1, 1), done)
        self.assertNotIn(m.session_key(2024, 1, 2), done)
        self.assertIn(m.session_key(2024, 1, 3), done)
        self.assertIn(m.session_key(2024, 1, 4), done)
        self.assertIn(m.session_key(2024, 1, 5), done)

    def test_state_written_to_disk(self):
        """done_set must be persisted to the state file after each ok/not_found."""
        import cache_fastf1_data as m
        state_path = os.path.join(self.tmpdir, "progress.json")
        side = self._make_fetch_results(["ok", "not_found"])
        with patch.object(m, "STATE_FILE", state_path):
            with patch.object(m, "fetch_session", side_effect=side):
                done = set()
                m.process_round(2024, 1, "Bahrain", done)
        # File must exist and contain at least the ok session
        self.assertTrue(os.path.exists(state_path))
        with open(state_path) as f:
            saved = set(json.load(f))
        self.assertIn(m.session_key(2024, 1, 1), saved)
        # not_found marker must also be persisted
        self.assertIn(m.session_key(2024, 1, 2), saved)


# ---------------------------------------------------------------------------
# Tests for cache_year pending detection
# ---------------------------------------------------------------------------

class TestCacheYear(unittest.TestCase):

    def _make_schedule_row(self, round_num, name):
        return {"RoundNumber": round_num, "EventName": name}

    def _fake_schedule(self, rounds):
        """Return a DataFrame-like object from a list of (round_num, name)."""
        import pandas as pd
        return pd.DataFrame([
            {"RoundNumber": rn, "EventName": name}
            for rn, name in rounds
        ])

    def test_fully_cached_year_skips_all_rounds(self):
        import cache_fastf1_data as m
        rounds = [(1, "Bahrain"), (2, "Saudi Arabia")]
        done = set()
        # Fill in all MAX_SESSIONS keys for both rounds
        for rn, _ in rounds:
            for i in range(1, m.MAX_SESSIONS + 1):
                done.add(m.session_key(2024, rn, i))

        sched = self._fake_schedule(rounds)
        with patch.object(m.fastf1, "get_event_schedule", return_value=sched):
            with patch.object(m, "process_round") as mock_pr:
                m.cache_year(2024, done)
        # No rounds should be submitted to process_round
        mock_pr.assert_not_called()

    def test_pending_round_without_not_found_fix(self):
        """
        Before the fix: a round where session 5 returned not_found would keep
        re-appearing as pending because key 5 was never added to done_set.
        After the fix: session 5 key IS added, so the round is correctly done.
        """
        import cache_fastf1_data as m
        rounds = [(1, "Bahrain")]
        done = set()
        # Simulate the fixed state: sessions 1-4 ok, 5 is not_found marker
        for i in range(1, 6):  # all 5 keys present after the fix
            done.add(m.session_key(2024, 1, i))

        sched = self._fake_schedule(rounds)
        with patch.object(m.fastf1, "get_event_schedule", return_value=sched):
            with patch.object(m, "process_round") as mock_pr:
                m.cache_year(2024, done)
        # With all 5 keys present, round must NOT be pending
        mock_pr.assert_not_called()

    def test_partially_cached_round_is_pending(self):
        """A round missing any session key must appear as pending."""
        import cache_fastf1_data as m
        rounds = [(1, "Bahrain")]
        done = set()
        # Only sessions 1-3 done (4 and 5 absent)
        for i in range(1, 4):
            done.add(m.session_key(2024, 1, i))

        sched = self._fake_schedule(rounds)
        with patch.object(m.fastf1, "get_event_schedule", return_value=sched):
            with patch.object(m, "process_round", return_value=(1, "Bahrain", 2)) as mock_pr:
                with patch.object(m, "WORKERS", 1):
                    m.cache_year(2024, done)
        mock_pr.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for load_state / save_state round-trip
# ---------------------------------------------------------------------------

class TestStateIO(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.tmpdir, "progress.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_roundtrip(self):
        import cache_fastf1_data as m
        original = {"2024_1_1", "2024_1_2", "2024_2_5"}
        with patch.object(m, "STATE_FILE", self.state_path):
            m.save_state(original)
            loaded = m.load_state()
        self.assertEqual(original, loaded)

    def test_load_missing_file_returns_empty(self):
        import cache_fastf1_data as m
        with patch.object(m, "STATE_FILE", self.state_path + "_nonexistent"):
            result = m.load_state()
        self.assertEqual(result, set())

    def test_save_is_atomic(self):
        """save_state must write a .tmp file then rename it - no partial writes."""
        import cache_fastf1_data as m
        with patch.object(m, "STATE_FILE", self.state_path):
            # Patch os.replace to verify the tmp file existed before the rename
            original_replace = os.replace
            tmp_existed = []

            def checking_replace(src, dst):
                tmp_existed.append(os.path.exists(src))
                return original_replace(src, dst)

            with patch("os.replace", side_effect=checking_replace):
                m.save_state({"2024_1_1"})

        self.assertTrue(tmp_existed[0], ".tmp file did not exist before os.replace")
        # After save, the .tmp file must be gone (renamed to STATE_FILE)
        self.assertFalse(os.path.exists(self.state_path + ".tmp"))


# ---------------------------------------------------------------------------
# Tests for data preprocessing helpers
# ---------------------------------------------------------------------------

class TestRaceGapPreprocessing(unittest.TestCase):

    def test_winner_gap_is_zero_even_with_total_race_time(self):
        from data_organized.prepocess_data import gap_to_winner_seconds
        self.assertEqual(
            gap_to_winner_seconds(1, "0 days 01:33:56.736000"),
            0.0,
        )

    def test_non_winner_time_is_already_gap(self):
        from data_organized.prepocess_data import gap_to_winner_seconds
        self.assertEqual(
            gap_to_winner_seconds(2, "0 days 00:00:11.987000"),
            11.987,
        )

    def test_missing_time_stays_missing_for_lapped_or_dnf_driver(self):
        import math
        import pandas as pd
        from data_organized.prepocess_data import gap_to_winner_seconds
        self.assertTrue(math.isnan(gap_to_winner_seconds(18, pd.NaT)))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
