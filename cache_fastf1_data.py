# Caches all FastF1 session data (FP1-3, Quali, Sprint, Race) from 2018-2026.
# FastF1 handles skip-if-cached automatically — just call session.load().

import fastf1
import logging
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("fastf1_cache.log", mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Suppress internal noise
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

CACHE_DIR = "discovery_raw_data"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEARS = range(2018, 2027)
MAX_SESSIONS = 5   # FP1, FP2/SQ, FP3/Sprint, Quali, Race
MAX_RETRIES = 3
RETRY_DELAY = 8    # seconds


def load_session(year, round_num, sess_idx):
    """Load one session; returns None if it doesn't exist or fails after retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(year, round_num, sess_idx)
            session.load(laps=True, telemetry=True, weather=True, messages=False)
            return session
        except ValueError:
            return None  # session index not valid for this event
        except Exception as exc:
            if attempt < MAX_RETRIES:
                log.warning("  Attempt %d/%d failed: %s — retrying in %ds", attempt, MAX_RETRIES, exc, RETRY_DELAY)
                time.sleep(RETRY_DELAY)
            else:
                log.error("  Failed after %d attempts: %s", MAX_RETRIES, exc)
                return None


def cache_year(year):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        log.error("Could not fetch %d schedule: %s", year, exc)
        return

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        log.info("  [R%02d] %s", round_num, event["EventName"])

        loaded = 0
        for idx in range(1, MAX_SESSIONS + 1):
            session = load_session(year, round_num, idx)
            if session is None:
                break
            log.info("    + %s", session.name)
            loaded += 1

        if loaded == 0:
            log.warning("    ! No sessions loaded")


def main():
    log.info("=== FastF1 cache: %d-%d ===", min(YEARS), max(YEARS))
    start = time.time()
    for year in YEARS:
        log.info("--- %d ---", year)
        cache_year(year)
    elapsed = time.time() - start
    log.info("Done in %.0fm %.0fs", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    main()
