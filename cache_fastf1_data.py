import fastf1
import logging
import time
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

CACHE_DIR = "discovery_raw_data"
STATE_FILE = os.path.join(CACHE_DIR, "cache_progress.json")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEARS = range(2018, 2027)
MAX_SESSIONS = 5

# Default to 1 worker: FastF1's internal rate limiter is not thread-safe under
# concurrent load (200 calls/h to ergast, 500/h to any API), so parallel workers
# race each other into the limit.  Set FF1_WORKERS=2 only when running locally
# with a fully warm HTTP cache (discovery_raw_data/fastf1_http_cache.sqlite).
WORKERS = int(os.environ.get("FF1_WORKERS", "1"))

# Back-off schedule for fastf1.RateLimitExceededError (seconds).
# FastF1 uses a 1-hour rolling window, so the worst case is waiting ~60 min.
RATE_LIMIT_BACKOFFS = [60, 300, 1200, 3600]

# Back-off schedule for other transient errors (seconds).
RETRY_DELAYS = [15, 45, 120]

state_lock = threading.Lock()


def load_state():
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_state(done: set):
    # atomic write so a crash mid-save doesn't corrupt the file
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done), f, indent=2)
    os.replace(tmp, STATE_FILE)


def session_key(year, round_num, idx):
    return f"{year}_{round_num}_{idx}"


def fetch_session(year, round_num, idx):
    """
    Download and cache one session. Returns (status, session):
      ('ok', session)     - loaded successfully
      ('not_found', None) - no session at this index for this event
      ('failed', None)    - unrecoverable after all retries

    RateLimitExceededError retries are tracked separately so they don't consume
    the transient-error budget - FastF1's rolling window can require a full hour
    to reset, and we'd rather wait than permanently skip a session.
    """
    rl_attempt = 0   # rate-limit retry counter
    tx_attempt = 0   # transient-error retry counter

    while True:
        try:
            session = fastf1.get_session(year, round_num, idx)
            session.load(laps=True, telemetry=True, weather=True, messages=False)
            return "ok", session

        except ValueError:
            # FastF1 raises ValueError when the session index is out of range
            return "not_found", None

        except fastf1.RateLimitExceededError as exc:
            # FastF1 enforces 200 calls/h (ergast) and 500 calls/h (any API).
            # Back off through the schedule; give up only after the longest wait.
            if rl_attempt >= len(RATE_LIMIT_BACKOFFS):
                log.error(
                    "  [%d R%02d S%d] rate-limit retries exhausted: %s",
                    year, round_num, idx, exc,
                )
                return "failed", None
            wait = RATE_LIMIT_BACKOFFS[rl_attempt]
            log.warning(
                "  [%d R%02d S%d] rate-limited (backoff %d/%d) - sleeping %ds",
                year, round_num, idx,
                rl_attempt + 1, len(RATE_LIMIT_BACKOFFS), wait,
            )
            time.sleep(wait)
            rl_attempt += 1

        except Exception as exc:
            if tx_attempt >= len(RETRY_DELAYS):
                log.error(
                    "  [%d R%02d S%d] gave up after %d attempts: %s",
                    year, round_num, idx, tx_attempt, exc,
                )
                return "failed", None
            delay = RETRY_DELAYS[tx_attempt]
            log.warning(
                "  [%d R%02d S%d] attempt %d failed: %s - retry in %ds",
                year, round_num, idx, tx_attempt + 1, exc, delay,
            )
            time.sleep(delay)
            tx_attempt += 1


def process_round(year, round_num, event_name, done_set):
    """Load all sessions for one round, skipping anything already in done_set."""
    loaded = 0
    for idx in range(1, MAX_SESSIONS + 1):
        key = session_key(year, round_num, idx)

        with state_lock:
            if key in done_set:
                continue

        status, session = fetch_session(year, round_num, idx)

        if status == "not_found":
            # idx and all higher indices don't exist for this event.
            # Mark every remaining index in done_set so cache_year won't keep
            # re-queuing this round just because some keys are missing.
            with state_lock:
                for mark in range(idx, MAX_SESSIONS + 1):
                    done_set.add(session_key(year, round_num, mark))
                save_state(done_set)
            break

        if status == "failed":
            # Transient error after all retries - skip this session but keep
            # going so a single bad session doesn't block the whole round.
            log.warning(
                "  [%d R%02d S%d] skipped after repeated errors - will retry next run",
                year, round_num, idx,
            )
            continue

        # status == "ok"
        log.info("  [%d R%02d] + %s", year, round_num, session.name)
        with state_lock:
            done_set.add(key)
            save_state(done_set)
        loaded += 1

    return round_num, event_name, loaded


def cache_year(year, done_set):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        log.error("Could not fetch %d schedule: %s", year, exc)
        return

    rounds = [(int(r["RoundNumber"]), r["EventName"]) for _, r in schedule.iterrows()]

    # A round is pending if any of its session keys is absent from done_set.
    # Because process_round now writes not_found keys for missing sessions,
    # every completed round will have all MAX_SESSIONS keys present.
    pending = []
    for round_num, name in rounds:
        keys = [session_key(year, round_num, i) for i in range(1, MAX_SESSIONS + 1)]
        with state_lock:
            if not all(k in done_set for k in keys):
                pending.append((round_num, name))

    if not pending:
        log.info("  %d fully cached, nothing to do", year)
        return

    log.info("  %d: %d rounds pending", year, len(pending))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_round, year, rn, name, done_set): (rn, name)
            for rn, name in pending
        }
        for fut in as_completed(futures):
            rn, name = futures[fut]
            try:
                _, _, loaded = fut.result()
                log.info("  [%d R%02d] %s - %d sessions", year, rn, name, loaded)
            except Exception as exc:
                log.error("  [%d R%02d] %s crashed: %s", year, rn, name, exc)


def main():
    done_set = load_state()
    log.info(
        "=== FastF1 cache: %d-%d | %d sessions already done ===",
        min(YEARS), max(YEARS), len(done_set),
    )

    start = time.time()
    for year in YEARS:
        log.info("--- %d ---", year)
        cache_year(year, done_set)

    elapsed = time.time() - start
    log.info("Done in %.0fm %.0fs", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    main()
