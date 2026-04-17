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
RETRY_DELAYS = [5, 15, 45]
WORKERS = 4  # parallel rounds per year — don't go too high or the API gets unhappy

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
        json.dump(sorted(done), f)
    os.replace(tmp, STATE_FILE)


def session_key(year, round_num, idx):
    return f"{year}_{round_num}_{idx}"


def fetch_session(year, round_num, idx):
    """
    Download and cache one session.  Returns a (status, session) tuple:
      ('ok', session)     — loaded successfully
      ('not_found', None) — this session index doesn't exist for this event
      ('failed', None)    — network/parse error after all retries; try again next run

    Keeping these three outcomes distinct is critical:
    - 'not_found' means we should stop trying higher session indices for this round.
    - 'failed'    means a transient error; we skip this session but keep going so
                  we don't miss sessions 3-5 just because session 2 had a hiccup.
    """
    for attempt, delay in enumerate(RETRY_DELAYS + [None], 1):
        try:
            session = fastf1.get_session(year, round_num, idx)
            session.load(laps=True, telemetry=True, weather=True, messages=False)
            return "ok", session
        except ValueError:
            # session index out of range for this event — no point retrying
            return "not_found", None
        except Exception as exc:
            if delay is None:
                log.error(
                    "  [%d R%02d S%d] gave up after %d attempts: %s",
                    year, round_num, idx, attempt - 1, exc,
                )
                return "failed", None
            log.warning(
                "  [%d R%02d S%d] attempt %d failed: %s — retry in %ds",
                year, round_num, idx, attempt, exc, delay,
            )
            time.sleep(delay)


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
            # No session at this index — stop trying higher indices for this round
            break

        if status == "failed":
            # Transient error — skip and continue so later sessions still get tried
            log.warning(
                "  [%d R%02d S%d] skipped after repeated errors — will retry on next run",
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

    # figure out which rounds still need work
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
                log.info("  [%d R%02d] %s — %d sessions", year, rn, name, loaded)
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
