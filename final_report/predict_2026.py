"""Retrain on recent seasons and project the 2026 F1 season.

This script intentionally retrains the ensemble instead of trying to fine-tune
old model files. Tree models need to see the full training distribution at once,
and keeping the 2026 artifacts in models_2026/ avoids overwriting the validation
models used elsewhere in the report.

Run from the project root:
    python final_report/predict_2026.py
"""

import os
import sys
import warnings
import logging

import fastf1
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless rendering - same reason as predict_2025.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

import xgboost as xgb

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pull in everything we need from mach_learning.py
# keeping the training logic there means both predict scripts share the same
# underlying functions and we only have to fix a bug in one place
from mach_learning import (
    FEATURE_COLS,
    SEED,
    POINTS_MAP,
    ensemble_predict,
    rank_predictions_per_race,
    _rank_normalize,
    compute_race_groups,
    train_xgboost_ranker,
    tune_xgboost,
    tune_random_forest_winner,
    tune_random_forest_top10,
    tune_extratrees,
    train_extratrees,
    winner_accuracy,
    podium_accuracy,
    top10_set_accuracy,
    average_position_mae,
    leave_one_season_out_validation,
    bayesian_model_averaging_weights,
)

# ── paths and config ──────────────────────────────────────────────────────────

DATA_DIR     = "data_organized"
MODEL_DIR    = os.path.join(DATA_DIR, "models_2026")  # separate from 2025 models - important
PLOT_DIR     = os.path.join("final_report", "plots")
CACHE_DIR    = "discovery_raw_data"

TRAIN_FILE   = os.path.join(DATA_DIR, "ml_training_features.csv")  # 2022-2024 from feature_engineering.py
FEAT_2025    = os.path.join(DATA_DIR, "ml_2025_features.csv")       # 2025 features + actuals (if available)

TRAIN_2026   = os.path.join(DATA_DIR, "2026_training_data.csv")     # combined 2022-2025 training
FEAT_2026    = os.path.join(DATA_DIR, "2026_features.csv")           # 2026 pre-race features
PRED_2026    = os.path.join(DATA_DIR, "2026_predictions.csv")        # our 2026 predictions
METR_2026    = os.path.join(DATA_DIR, "2026_race_metrics.csv")       # accuracy metrics for completed races
WDC_2026     = os.path.join(DATA_DIR, "2026_wdc_projection.csv")     # WDC standings projection

PREDICT_YEAR = 2026
TRAIN_YEARS  = [2022, 2023, 2024, 2025]

# same colour scheme as predict_2025.py - visual consistency across the project
# it would look weird if the 2025 plots were dark and the 2026 plots were white
F1_DARK  = "#15151E"
F1_RED   = "#E8002D"
F1_BLUE  = "#3A86FF"
F1_WHITE = "#FFFFFF"
F1_GREY  = "#38383F"
F1_AMBER = "#FFBE0B"
F1_GREEN = "#00C853"

fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)


# ── build combined 2022-2025 training data ────────────────────────────────────

def build_combined_training_data():
    # combine the 2022-2024 training data with whatever 2025 races are in the cache
    # adding 2025 gives us more examples of recent-season driver behaviour,
    # which matters because teams and drivers change over years
    #
    # one wrinkle: at the start of the 2025 season this file might only have
    # a few races. We still include it because even partial data is better than nothing
    # for recent form features. The model just has fewer 2025 examples to learn from.
    #
    # ref: why more training data helps: https://developers.google.com/machine-learning/data-prep/construct/collect/data-size-quality

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"\n[!] 2022-2024 training file not found: {TRAIN_FILE}\n"
            f"    Run  python final_report/feature_engineering.py  first."
        )
    df_train = pd.read_csv(TRAIN_FILE)
    print(f"  ✓ 2022-2024 rows: {len(df_train)}")

    # load whatever 2025 completed races we have
    df_2025 = pd.DataFrame()
    if os.path.exists(FEAT_2025):
        df_raw = pd.read_csv(FEAT_2025)
        if "finish_position" in df_raw.columns:
            df_2025 = df_raw.dropna(subset=["finish_position"]).copy()
        if not df_2025.empty:
            print(f"  ✓ 2025 completed rows: {len(df_2025)} ({df_2025['Round'].nunique()} races)")
        else:
            print("  [!] 2025 feature file exists but no completed races found in it")
    else:
        print(f"  [!] 2025 features not found at {FEAT_2025} - training on 2022-2024 only")
        print("      Run feature_engineering.py with PREDICT_YEAR=2025 to fix this")

    # concatenate - if df_2025 is empty this just returns df_train unchanged
    if not df_2025.empty:
        combined = pd.concat([df_train, df_2025], ignore_index=True)
    else:
        combined = df_train.copy()

    n_seasons = combined["Year"].nunique()
    n_races   = combined[["Year", "Round"]].drop_duplicates().shape[0]
    print(f"  ✓ Combined: {len(combined)} rows, {n_seasons} seasons, {n_races} race weekends")

    combined.to_csv(TRAIN_2026, index=False)
    print(f"  → Saved combined training data: {TRAIN_2026}")

    # extract what we need for model training
    meta_cols = ["Year", "Round", "EventName", "Circuit", "Driver", "Team"]
    df_meta   = combined[[c for c in meta_cols if c in combined.columns]].copy()
    X_raw     = combined[FEATURE_COLS].copy().astype(float)

    # fit a new imputer on the combined 2022-2025 training data
    # can't reuse the 2022-2024 imputer because the feature distribution may have shifted
    # ref: https://scikit-learn.org/stable/modules/impute.html
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURE_COLS)

    y_pos   = combined["finish_position"].astype(float) if "finish_position" in combined else None
    y_win   = combined["is_winner"].astype(float)       if "is_winner"   in combined else None
    y_top10 = combined["is_top10"].astype(float)        if "is_top10"    in combined else None

    # drop any rows where targets are NaN - can't train on those
    valid   = y_pos.notna() & y_win.notna() & y_top10.notna()
    X       = X[valid].reset_index(drop=True)
    y_pos   = y_pos[valid].reset_index(drop=True)
    y_win   = y_win[valid].reset_index(drop=True)
    y_top10 = y_top10[valid].reset_index(drop=True)
    df_meta = df_meta[valid].reset_index(drop=True)
    df_full = combined[valid].reset_index(drop=True)

    print(f"  ✓ After dropping NaN targets: {len(X)} rows ready for training")
    return X, y_pos, y_win, y_top10, df_meta, df_full, imputer


# ── model training (2022-2025 data) ──────────────────────────────────────────

def train_models_2026(X_train, y_pos, y_win, y_top10, df_meta):
    # full training pipeline: LOSO validation to get Bayesian weights, then
    # tune and train all three models on the complete 2022-2025 dataset
    #
    # LOSO on 4 seasons (2022, 2023, 2024, 2025) takes around 10-15 minutes
    # with the tuning enabled. We run it first because we need the per-model
    # correct-winner counts to compute the Bayesian Model Averaging weights.
    # ref: https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging
    #
    # note: leave_one_season_out_validation only needs Year/Round/EventName/Driver/Team
    # from df_meta - it does its own feature extraction internally from X_train
    print("\n  Running LOSO cross-validation on 2022-2025 (computing Bayesian weights)...")
    val_results, bma_weights, cal_temperature, meta_model = leave_one_season_out_validation(
        df_meta,
        X_train,
        y_pos, y_win, y_top10
    )

    print("\n  Tuning XGBoost regressor (50 RandomizedSearch iterations)...")
    xgb_final, xgb_best = tune_xgboost(X_train, y_pos.fillna(20), n_iter=50)
    print("  ✓ XGBoost regressor done")

    print("\n  Tuning Random Forest winner classifier (40 iterations)...")
    rf_win_final, _ = tune_random_forest_winner(X_train, y_win, n_iter=40)
    print("  ✓ Random Forest (winner) done")

    print("\n  Tuning Random Forest top-10 classifier (40 iterations)...")
    rf_top10_final, _ = tune_random_forest_top10(X_train, y_top10, n_iter=40)
    print("  ✓ Random Forest (top-10) done")

    print("\n  Tuning ExtraTrees regressor (40 iterations)...")
    et_final, _ = tune_extratrees(X_train, y_pos.fillna(20), n_iter=40)
    print("  ✓ ExtraTrees done")

    # XGBRanker needs data sorted by (Year, Round) for the group structure to work
    # the groups tell the ranker which rows belong to the same race and should be ranked together
    # getting this wrong was one of the first bugs we hit - the ranker was treating all 4000+ rows
    # as one giant ranking problem which made no sense for race-by-race prediction
    # ref: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
    print("\n  Training XGBoost ranker (rank:ndcg)...")
    sort_idx      = df_meta.sort_values(["Year", "Round"]).index
    X_sorted      = X_train.loc[sort_idx].reset_index(drop=True)
    y_sorted      = y_pos.loc[sort_idx].reset_index(drop=True)
    meta_sorted   = df_meta.loc[sort_idx].reset_index(drop=True)
    ranker_groups = compute_race_groups(meta_sorted)
    xgb_ranker    = train_xgboost_ranker(X_sorted, y_sorted, ranker_groups, xgb_best)
    print("  ✓ XGBoost ranker done")

    return xgb_final, xgb_ranker, rf_win_final, rf_top10_final, et_final, bma_weights, cal_temperature, meta_model


def _train_models_without_loso(X_train, y_pos, y_win, y_top10, df_meta):
    # faster training path that skips LOSO - use this on the first run to avoid
    # waiting 15 minutes for validation to finish before seeing any predictions
    #
    # BMA weights are fixed at 40/40/20 here instead of being learned from data
    # that's less principled but not catastrophically wrong - empirically XGBoost and
    # the ranker tend to be roughly equally good so 40/40 is a reasonable starting point
    # and RF is clearly the weakest of the three so 20 makes sense for that
    #
    # if you want the Bayesian-computed weights, call train_models_2026() instead
    # (just be prepared to wait a while)
    print("\n  (Skipping LOSO - using fixed BMA weights for speed)")
    bma_weights = {"w_xgb": 0.35, "w_ranker": 0.35, "w_rf": 0.15, "w_et": 0.15}

    print("  Tuning XGBoost regressor (50 iterations)...")
    xgb_final, xgb_best = tune_xgboost(X_train, y_pos.fillna(20), n_iter=50)
    print("  ✓ XGBoost done")

    print("  Tuning Random Forest winner (40 iterations)...")
    rf_win_final, _ = tune_random_forest_winner(X_train, y_win, n_iter=40)
    print("  ✓ RF winner done")

    print("  Tuning Random Forest top-10 (40 iterations)...")
    rf_top10_final, _ = tune_random_forest_top10(X_train, y_top10, n_iter=40)
    print("  ✓ RF top-10 done")

    print("  Tuning ExtraTrees regressor (40 iterations)...")
    et_final, _ = tune_extratrees(X_train, y_pos.fillna(20), n_iter=40)
    print("  ✓ ExtraTrees done")

    print("  Training XGBoost ranker...")
    sort_idx      = df_meta.sort_values(["Year", "Round"]).index
    X_sorted      = X_train.loc[sort_idx].reset_index(drop=True)
    y_sorted      = y_pos.loc[sort_idx].reset_index(drop=True)
    meta_sorted   = df_meta.loc[sort_idx].reset_index(drop=True)
    ranker_groups = compute_race_groups(meta_sorted)
    xgb_ranker    = train_xgboost_ranker(X_sorted, y_sorted, ranker_groups, xgb_best)
    print("  ✓ Ranker done")

    # no LOSO means no meta-model; return None so callers skip stacking
    return xgb_final, xgb_ranker, rf_win_final, rf_top10_final, et_final, bma_weights, 1.0, None


def save_models_2026(xgb_final, xgb_ranker, rf_win, rf_top10, et_model, imputer,
                     bma_weights, cal_temperature=1.0, meta_model=None):
    # save all model artefacts so we don't have to retrain every time we run predictions
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_final,      os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(xgb_ranker,     os.path.join(MODEL_DIR, "xgb_ranker.joblib"))
    joblib.dump(rf_win,         os.path.join(MODEL_DIR, "rf_win_model.joblib"))
    joblib.dump(rf_top10,       os.path.join(MODEL_DIR, "rf_top10_model.joblib"))
    if et_model is not None:
        joblib.dump(et_model,   os.path.join(MODEL_DIR, "et_model.joblib"))
    joblib.dump(imputer,        os.path.join(MODEL_DIR, "feature_imputer.joblib"))
    joblib.dump(bma_weights,    os.path.join(MODEL_DIR, "bma_weights.joblib"))
    joblib.dump(cal_temperature, os.path.join(MODEL_DIR, "cal_temperature.joblib"))
    if meta_model is not None:
        joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.joblib"))
    print(f"  → All models saved to {MODEL_DIR}/")


def load_models_2026():
    # try to load cached 2026 models - if any required file is missing, signal caller to retrain
    required = {
        "xgb":      os.path.join(MODEL_DIR, "xgb_model.joblib"),
        "ranker":   os.path.join(MODEL_DIR, "xgb_ranker.joblib"),
        "rf_win":   os.path.join(MODEL_DIR, "rf_win_model.joblib"),
        "rf_top10": os.path.join(MODEL_DIR, "rf_top10_model.joblib"),
        "imputer":  os.path.join(MODEL_DIR, "feature_imputer.joblib"),
    }
    if any(not os.path.exists(p) for p in required.values()):
        return None, None, None, None, None, None, None, None, None

    bma_path    = os.path.join(MODEL_DIR, "bma_weights.joblib")
    bma_weights = joblib.load(bma_path) if os.path.exists(bma_path) else None

    cal_path    = os.path.join(MODEL_DIR, "cal_temperature.joblib")
    cal_temperature = joblib.load(cal_path) if os.path.exists(cal_path) else 1.0

    meta_path   = os.path.join(MODEL_DIR, "meta_model.joblib")
    meta_model  = joblib.load(meta_path) if os.path.exists(meta_path) else None

    et_path     = os.path.join(MODEL_DIR, "et_model.joblib")
    et_model    = joblib.load(et_path) if os.path.exists(et_path) else None

    return (
        joblib.load(required["xgb"]),
        joblib.load(required["ranker"]),
        joblib.load(required["rf_win"]),
        joblib.load(required["rf_top10"]),
        et_model,
        joblib.load(required["imputer"]),
        bma_weights,
        cal_temperature,
        meta_model,
    )


# ── 2026 feature building ─────────────────────────────────────────────────────

def build_2026_features(imputer):
    # build pre-race features for 2026 races using the same feature_engineering pipeline
    # this runs warmup years 2019-2025 first to populate rolling stats correctly,
    # then extracts features for 2026 race weekends
    #
    # because this takes a long time (processing 7+ years of FastF1 sessions),
    # we cache the result to FEAT_2026 and skip rebuilding if it already exists
    # delete the file manually if you need to force a fresh build (e.g. new races added)

    if os.path.exists(FEAT_2026):
        print(f"  ✓ Loading cached 2026 features from {FEAT_2026}")
        df = pd.read_csv(FEAT_2026)
    else:
        print("  Building 2026 features from FastF1 cache...")
        print("  (Processes warmup data 2019-2025 + 2026 sessions - takes several minutes)")
        try:
            from feature_engineering import build_features as fe_build
        except ImportError:
            raise ImportError(
                "\n[!] Cannot import feature_engineering.py.\n"
                "    Make sure you're running from the project root:\n"
                "      python final_report/predict_2026.py"
            )
        df = fe_build([PREDICT_YEAR], include_race_target=False)
        df.to_csv(FEAT_2026, index=False)
        print(f"  → Saved 2026 features: {FEAT_2026}")

    if df.empty:
        raise RuntimeError(
            f"\n[!] No 2026 features built - cache is probably missing 2026 sessions.\n"
            f"    Run cache_fastf1_data.py with YEARS=[2026] and try again.\n"
            f"    Cache dir: {CACHE_DIR}/"
        )

    # apply the imputer fitted on 2022-2025 training data
    # same rule as predict_2025.py: transform only, do not refit here
    X_raw = df.reindex(columns=FEATURE_COLS).astype(float)
    X = pd.DataFrame(imputer.transform(X_raw), columns=FEATURE_COLS, index=df.index)

    n_races   = df["Round"].nunique()
    n_drivers = df["Driver"].nunique()
    print(f"  ✓ 2026 features: {len(df)} rows, {n_races} races, {n_drivers} drivers")
    return X, df


# ── actual 2026 results (for validation of completed races) ───────────────────

def load_actual_2026_results():
    # load real results for any 2026 races that are already in the FastF1 cache
    # at the start of the season this will return an empty DataFrame (no races done)
    # as the season progresses more races will appear here for validation
    rows = []
    try:
        schedule = fastf1.get_event_schedule(PREDICT_YEAR, include_testing=False)
    except Exception as e:
        print(f"  [!] Cannot load 2026 schedule: {e}")
        return pd.DataFrame()

    for _, event in schedule.iterrows():
        rnd  = int(event["RoundNumber"])
        name = str(event["EventName"])
        try:
            session = fastf1.get_session(PREDICT_YEAR, rnd, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            for _, row in session.results.iterrows():
                abbr = str(row.get("Abbreviation", ""))
                pos  = row.get("Position", np.nan)
                if abbr and pd.notna(pos):
                    rows.append({
                        "Round":           rnd,
                        "EventName":       name,
                        "Driver":          abbr,
                        "actual_position": float(pos),
                    })
        except Exception:
            pass  # race not in cache yet

    df = pd.DataFrame(rows)
    n_done = df["Round"].nunique() if not df.empty else 0
    print(f"  ✓ Actual 2026 results: {n_done} completed races found in cache")
    return df


# ── WDC championship projection ───────────────────────────────────────────────

def pos_to_points(pos):
    # convert predicted finishing position to championship points
    # standard 2026 F1 points system (assuming unchanged from 2025)
    # ref: https://en.wikipedia.org/wiki/List_of_Formula_One_World_Championship_points_scoring_systems
    try:
        return POINTS_MAP.get(int(pos), 0)
    except (ValueError, TypeError):
        return 0


def project_wdc(pred_df):
    # sum up predicted points across all races to get a season projection
    # this is a pretty naive approach - it doesn't account for reliability
    # differences, strategic upsets, or the fact that predictions for later
    # races are inherently less confident than early-season ones
    # but it gives a directionally useful picture of who should be champion
    # if the model's race-by-race calls are roughly right

    pts = (
        pred_df.groupby(["Driver", "Team"])["PredictedPosition"]
        .apply(lambda s: sum(pos_to_points(p) for p in s))
        .reset_index()
        .rename(columns={"PredictedPosition": "predicted_points"})
        .sort_values("predicted_points", ascending=False)
        .reset_index(drop=True)
    )
    pts["predicted_rank"] = pts.index + 1

    # if we have actual results for completed races, add actual points scored so far
    # this lets us compare predicted vs actual championship position mid-season
    if "actual_position" in pred_df.columns:
        act_pts = (
            pred_df.dropna(subset=["actual_position"])
            .groupby("Driver")["actual_position"]
            .apply(lambda s: sum(pos_to_points(p) for p in s))
            .reset_index()
            .rename(columns={"actual_position": "actual_points_sofar"})
        )
        pts = pts.merge(act_pts, on="Driver", how="left")

    return pts


def write_wdc_predictions_txt(wdc_df, n_predicted, n_total, out_dir):
    # write a plain-text summary of the 2026 WDC predictions so it can be
    # included in the report without needing to open a CSV or plot
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "2026_wdc_predictions.txt")

    # full driver name lookup — three-letter codes from the feature data map to full names
    full_names = {
        "VER": "Max Verstappen",     "NOR": "Lando Norris",
        "LEC": "Charles Leclerc",    "PIA": "Oscar Piastri",
        "SAI": "Carlos Sainz",       "RUS": "George Russell",
        "HAM": "Lewis Hamilton",     "ALO": "Fernando Alonso",
        "STR": "Lance Stroll",       "GAS": "Pierre Gasly",
        "OCO": "Esteban Ocon",       "TSU": "Yuki Tsunoda",
        "ALB": "Alexander Albon",    "BOT": "Valtteri Bottas",
        "ZHO": "Guanyu Zhou",        "MAG": "Kevin Magnussen",
        "HUL": "Nico Hülkenberg",    "LAW": "Liam Lawson",
        "BEA": "Oliver Bearman",     "ANT": "Kimi Antonelli",
        "HAD": "Isack Hadjar",       "DOO": "Jack Doohan",
        "SAR": "Logan Sargeant",     "COL": "Franco Colapinto",
        "PER": "Sergio Perez",       "RIC": "Daniel Ricciardo",
    }

    champion_row = wdc_df.iloc[0]
    champion_abbr = champion_row["Driver"]
    champion_name = full_names.get(champion_abbr, champion_abbr)
    champion_team = str(champion_row.get("Team", ""))
    champion_pts  = int(champion_row["predicted_points"])

    lines = [
        "=" * 62,
        "  2026 F1 WORLD CHAMPIONSHIP — PREDICTED FINAL STANDINGS",
        "=" * 62,
        f"  Races predicted : {n_predicted} of {n_total}",
        f"  Model           : 3-model BMA ensemble (XGBoost Regressor +",
        f"                    XGBoost Ranker + Random Forest + ExtraTrees)",
        f"  Training data   : 2022–2025 seasons",
        "",
        "  NOTE: 2026 introduces the largest regulation change since 2014",
        "  (active aerodynamics + all-new hybrid power units). Historical",
        "  team performance rankings may not carry forward. These",
        "  predictions reflect driver-skill and circuit-history signals;",
        "  treat them as directional, not definitive.",
        "",
        "─" * 62,
        f"  PREDICTED WORLD CHAMPION:",
        f"    {champion_name} ({champion_team})  —  {champion_pts} projected points",
        "─" * 62,
        "",
        f"  {'Rank':>4}  {'Driver':<22}  {'Team':<28}  {'Pred Pts':>9}",
        f"  {'────':>4}  {'──────────────────────':22}  {'────────────────────────────':28}  {'─────────':>9}",
    ]

    for _, row in wdc_df.iterrows():
        abbr      = row["Driver"]
        full      = full_names.get(abbr, abbr)
        team      = str(row.get("Team", ""))[:28]
        pts       = int(row["predicted_points"])
        rank      = int(row["predicted_rank"])
        act_note  = ""
        if "actual_points_sofar" in row and pd.notna(row.get("actual_points_sofar")):
            act_note = f"  (actual so far: {int(row['actual_points_sofar'])} pts)"
        lines.append(
            f"  {rank:>4}  {full:<22}  {team:<28}  {pts:>9}{act_note}"
        )

    lines += [
        "",
        "=" * 62,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  → WDC predictions text: {path}")
    return path


def print_wdc_projection(wdc_df, n_predicted, n_total):
    # console printout of the standings table - good for a quick sanity check
    print(f"\n  2026 WDC Projection  ({n_predicted}/{n_total} races predicted)")
    print(f"  {'Rank':>4}  {'Driver':<6}  {'Team':<30}  {'Pred Pts':>8}")
    print(f"  {'────':>4}  {'──────':<6}  {'──────────────────────────────':<30}  {'────────':>8}")
    for _, row in wdc_df.head(20).iterrows():
        act_str = ""
        if "actual_points_sofar" in row and pd.notna(row.get("actual_points_sofar")):
            act_str = f"  (actual so far: {int(row['actual_points_sofar'])} pts)"
        print(f"  {int(row['predicted_rank']):>4}  "
              f"{row['Driver']:<6}  "
              f"{str(row['Team'])[:30]:<30}  "
              f"{int(row['predicted_points']):>8}"
              f"{act_str}")


# ── per-race accuracy metrics ─────────────────────────────────────────────────

def compute_per_race_metrics_2026(pred_df):
    # exact same logic as predict_2025.py - no reason to change it
    # returns empty DataFrame if no races are completed yet which is expected early in the season
    if "actual_position" not in pred_df.columns:
        return pd.DataFrame()

    completed = pred_df.dropna(subset=["actual_position"])
    if completed.empty:
        return pd.DataFrame()

    rows = []
    for (year, rnd), g in completed.groupby(["Year", "Round"]):
        g    = g.reset_index(drop=True)
        meta = g[["Year", "Round", "EventName", "Driver", "Team"]]
        pred = g["PredictedPosition"].astype(float)
        true = g["actual_position"].astype(float)

        w_acc,  _, _      = winner_accuracy(meta, pred, true)
        pod_ex, pod_ol, _ = podium_accuracy(meta, pred, true)
        jaccard           = top10_set_accuracy(meta, pred, true)
        mae               = average_position_mae(meta, pred, true)

        rows.append({
            "Year":           year,
            "Round":          rnd,
            "EventName":      g.iloc[0]["EventName"],
            "winner_correct": w_acc,
            "podium_overlap": pod_ol,
            "top10_jaccard":  jaccard,
            "position_mae":   mae,
        })
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_wdc_projection(wdc_df, n_races, out_dir):
    # horizontal bar chart showing projected championship points for top 15 drivers
    # if actual points are available (completed races), shown as a gold overlay
    # the regulation caveat is annotated in the corner - felt important to make it visible
    # rather than burying it in comments nobody reads
    df = wdc_df.head(15).copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)
    fig.suptitle(
        f"2026 WDC Projection  ·  {n_races} races predicted",
        fontsize=13, color=F1_WHITE, fontweight="bold",
    )

    # gradient from red to blue across the championship order
    # top driver is warmest, 15th is coolest - visually intuitive
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(df)))

    bars = ax.barh(
        range(len(df)),
        df["predicted_points"].values[::-1],
        color=colors,
        edgecolor=F1_DARK,
        linewidth=0.4,
    )

    # overlay actual points as gold bars if we have completed race data
    # the gold bars effectively show "what really happened so far"
    # while the full bars show "what we think will happen by season end"
    if "actual_points_sofar" in df.columns and df["actual_points_sofar"].notna().any():
        ax.barh(
            range(len(df)),
            df["actual_points_sofar"].fillna(0).values[::-1],
            color=F1_AMBER,
            alpha=0.5,
            edgecolor="none",
            label="Actual pts (completed races)",
        )
        ax.legend(fontsize=9, facecolor=F1_DARK, edgecolor=F1_GREY, labelcolor=F1_WHITE)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"P{int(r['predicted_rank'])}  {r['Driver']}" for _, r in df.iloc[::-1].iterrows()],
        fontsize=10, color=F1_WHITE,
    )
    ax.set_xlabel("Predicted Championship Points", fontsize=10, color=F1_WHITE)
    ax.tick_params(colors=F1_WHITE)
    ax.grid(axis="x", alpha=0.2, color=F1_GREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(F1_GREY)

    # annotate point totals at end of each bar
    for bar, (_, row) in zip(bars[::-1], df.iloc[::-1].iterrows()):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(int(row["predicted_points"])),
                va="center", fontsize=8, color=F1_WHITE)

    # the caveat - 2026 regulations could make any of this wrong
    # we've seen regulation resets flip the competitive order entirely (2014 being the clearest example)
    ax.text(
        0.99, 0.01,
        "⚠  2026 active aero + new PUs: historical data may not transfer",
        transform=ax.transAxes, fontsize=7, color=F1_AMBER,
        ha="right", va="bottom",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "wdc_projection_2026.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_season_predictions(pred_df, out_dir, max_races=24):
    # grid showing predicted top-5 per race across the season
    # each column is one race, each row is a predicted position P1-P5
    # colour shows how close we were once actuals are available:
    #   green  = exact match
    #   yellow = off by 1-2 positions
    #   red    = wrong by more than 2
    #   grey   = future race, no actual yet
    #
    # this is the kind of visualisation that's immediately readable at a glance
    # which we liked because most of the other plots need some F1 context to interpret
    rounds   = sorted(pred_df["Round"].unique())[:max_races]
    pred_sub = pred_df[pred_df["Round"].isin(rounds)]

    n_races = len(rounds)
    fig, ax = plt.subplots(figsize=(max(12, n_races * 0.95), 5))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)
    fig.suptitle("2026 Season — Predicted Top-5 per Race",
                 fontsize=13, color=F1_WHITE, fontweight="bold")

    has_actuals = "actual_position" in pred_sub.columns

    for col_idx, rnd in enumerate(rounds):
        race  = pred_sub[pred_sub["Round"] == rnd].sort_values("PredictedPosition")
        label = race.iloc[0]["EventName"] if not race.empty else f"R{rnd}"
        label = label.replace(" Grand Prix", "").replace(" GP", "")

        for row_idx, (_, driver_row) in enumerate(race.head(5).iterrows()):
            pred_p = int(driver_row["PredictedPosition"])
            abbr   = driver_row["Driver"]

            # colour the cell based on prediction error if actuals exist
            if has_actuals and pd.notna(driver_row.get("actual_position")):
                actual_p = int(driver_row["actual_position"])
                err      = abs(pred_p - actual_p)
                bg = (0.0, 0.78, 0.32) if err == 0 else \
                     (1.0, 0.75, 0.04) if err <= 2 else \
                     (0.91, 0.0, 0.18)
            else:
                bg = (0.22, 0.22, 0.27)  # future race - neutral dark grey

            rect = plt.Rectangle(
                (col_idx - 0.45, 4 - row_idx - 0.45), 0.9, 0.9,
                color=bg, linewidth=0,
            )
            ax.add_patch(rect)
            ax.text(col_idx, 4 - row_idx, abbr,
                    ha="center", va="center", fontsize=7,
                    color=F1_WHITE, fontweight="bold")

        ax.text(col_idx, 5.15, label, ha="center", va="bottom",
                fontsize=6.5, color=F1_GREY, rotation=45)

    ax.set_xlim(-0.55, n_races - 0.45)
    ax.set_ylim(-0.55, 5.7)
    ax.set_xticks([])
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"P{i+1}" for i in range(4, -1, -1)],
                       fontsize=9, color=F1_WHITE)
    ax.grid(False)

    patches = [
        mpatches.Patch(color=(0.0, 0.78, 0.32), label="Exact match"),
        mpatches.Patch(color=(1.0, 0.75, 0.04), label="Off by 1-2"),
        mpatches.Patch(color=(0.91, 0.0, 0.18), label="Wrong (>±2)"),
        mpatches.Patch(color=(0.22, 0.22, 0.27), label="Future race"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              facecolor=F1_DARK, edgecolor=F1_GREY, labelcolor=F1_WHITE)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "season_predictions_2026.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_accuracy_timeline(metrics_df, out_dir):
    # for completed races: how does accuracy change as the season goes on?
    # we were curious whether the model gets worse as teams develop their cars
    # and the field order shifts - early 2022 predictions should be less accurate
    # than mid-season once the model's rolling-form features have caught up
    # ref for time series evaluation: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
    if metrics_df.empty:
        print("  [skip] no completed 2026 races for accuracy timeline")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)

    rounds = metrics_df["Round"].values
    jac    = metrics_df["top10_jaccard"].values
    win    = metrics_df["winner_correct"].values.astype(float)

    ax.plot(rounds, jac, color=F1_BLUE,  linewidth=2, marker="o", markersize=5, label="Top-10 Jaccard")
    ax.plot(rounds, win, color=F1_GREEN, linewidth=2, marker="s", markersize=5, label="Winner correct")

    # dashed reference lines at the season average
    ax.axhline(jac.mean(), color=F1_BLUE,  linewidth=1, linestyle="--", alpha=0.5)
    ax.axhline(win.mean(), color=F1_GREEN, linewidth=1, linestyle="--", alpha=0.5)

    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Round", color=F1_WHITE, fontsize=10)
    ax.set_ylabel("Score", color=F1_WHITE, fontsize=10)
    ax.set_title("2026 Prediction Accuracy — Completed Races",
                 color=F1_WHITE, fontsize=12, fontweight="bold")
    ax.tick_params(colors=F1_WHITE)
    ax.legend(fontsize=9, facecolor=F1_DARK, edgecolor=F1_GREY, labelcolor=F1_WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(F1_GREY)

    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_timeline_2026.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    print("=" * 60)
    print("  F1 2026 — Race Predictions & WDC Projection")
    print("=" * 60)

    # print the regulation caveat up front so it's impossible to miss
    # we added this after a teammate looked at the WDC output and thought it was
    # a confident prediction - it's really not, and we want that to be clear
    print("""
  NOTE: 2026 introduces the biggest regulation change since 2014:
    - Active aerodynamics (fully movable bodywork system)
    - Brand new hybrid power units from all manufacturers
    - Significant structural rule changes to car dimensions

  Historical team performance (2022-2025) may not carry forward at all.
  This model's driver-skill and circuit-history features should partially
  transfer, but car-level performance rankings are genuinely uncertain.
  The Bayesian prior on win probability leans on driver circuit history
  rather than pure team order, which adds some robustness - but there's
  only so much you can do when the cars are completely different.
  Treat 2026 outputs as directional, not predictive.
""")

    # step 1: combine 2022-2025 training data
    print("[1/6] Building combined 2022-2025 training dataset...")
    X_train, y_pos, y_win, y_top10, df_meta, df_full, imputer = \
        build_combined_training_data()

    # step 2: try to load cached models, retrain if they don't exist
    # on the first run this will train from scratch (~5-10 minutes without LOSO)
    # subsequent runs load from disk in seconds
    print("\n[2/6] Loading or training 2026 models...")
    xgb_model, xgb_ranker, rf_win, rf_top10, et_model, cached_imputer, bma_weights, cal_temperature, meta_model = \
        load_models_2026()

    if xgb_model is not None:
        print(f"  ✓ Loaded cached 2026 models from {MODEL_DIR}/")
        print("    Delete the models_2026/ directory to force a full retrain.")
        imputer         = cached_imputer
        bma_weights     = bma_weights or {"w_xgb": 0.35, "w_ranker": 0.35, "w_rf": 0.15, "w_et": 0.15}
        cal_temperature = cal_temperature if cal_temperature is not None else 1.0
        if bma_weights:
            print(f"  ✓ BMA weights: xgb={bma_weights.get('w_xgb', '?'):.3f}, "
                  f"ranker={bma_weights.get('w_ranker', '?'):.3f}, "
                  f"rf={bma_weights.get('w_rf', '?'):.3f}, "
                  f"et={bma_weights.get('w_et', '?'):.3f}")
        print(f"  ✓ Calibration temperature T={cal_temperature:.4f}")
        if meta_model is not None:
            print(f"  ✓ Stacking meta-model loaded")
        if et_model is None:
            print("  [!] No cached ET model found - retrain for best accuracy")
    else:
        print(f"  No cached models in {MODEL_DIR}/ - training from scratch")
        print("  (Using fast path - skipping LOSO. See train_models_2026() for full run.)")
        print("─" * 55)
        xgb_model, xgb_ranker, rf_win, rf_top10, et_model, bma_weights, cal_temperature, meta_model = \
            _train_models_without_loso(X_train, y_pos, y_win, y_top10, df_meta)
        save_models_2026(xgb_model, xgb_ranker, rf_win, rf_top10, et_model, imputer,
                         bma_weights, cal_temperature, meta_model)

    # step 3: build 2026 pre-race features from FastF1 cache
    print("\n[3/6] Building 2026 pre-race features...")
    X_2026, df_2026 = build_2026_features(imputer)

    # step 4: run predictions
    print("\n[4/6] Generating 2026 predictions (3-model ensemble + Bayesian prior)...")

    # sort by (Year, Round) for consistent ranker group ordering - same issue as predict_2025.py
    sort_order = df_2026.sort_values(["Year", "Round"]).index
    X_2026_s   = X_2026.loc[sort_order].reset_index(drop=True)
    df_2026_s  = df_2026.loc[sort_order].reset_index(drop=True)

    xgb_preds     = xgb_model.predict(X_2026_s)
    ranker_scores = xgb_ranker.predict(X_2026_s)
    rf_win_proba  = rf_win.predict_proba(X_2026_s)[:, 1]
    rf_top10_pred = rf_top10.predict(X_2026_s)
    et_preds      = et_model.predict(X_2026_s) if et_model is not None else None

    pred_df = ensemble_predict(
        xgb_preds,
        rf_win_proba,
        df_2026_s[["Year", "Round", "EventName", "Driver", "Team"]],
        ranker_raw_scores=ranker_scores,
        et_raw_pos=et_preds,
        ensemble_weights=bma_weights,
        X_features=X_2026_s,
        apply_bayesian_prior=True,
        cal_temperature=cal_temperature,
        meta_model=meta_model,
    )
    pred_df["rf_win_proba"]  = rf_win_proba
    pred_df["rf_top10_pred"] = rf_top10_pred
    pred_df["xgb_raw_pred"]  = xgb_preds

    n_races_predicted = pred_df["Round"].nunique()
    print(f"  ✓ Predictions generated: {n_races_predicted} races, {pred_df['Driver'].nunique()} drivers")

    # step 5: check against actual results for completed races
    print("\n[5/6] Loading actual 2026 race results (for any completed races)...")
    actual_df = load_actual_2026_results()

    if not actual_df.empty:
        pred_df = pred_df.merge(
            actual_df[["Round", "Driver", "actual_position"]],
            on=["Round", "Driver"],
            how="left",
        )
        metrics_df = compute_per_race_metrics_2026(pred_df)
        if not metrics_df.empty:
            metrics_df.to_csv(METR_2026, index=False)
            w_avg   = metrics_df["winner_correct"].mean()
            jac_avg = metrics_df["top10_jaccard"].mean()
            mae_avg = metrics_df["position_mae"].mean()
            print(f"\n  Validation ({len(metrics_df)} completed races):")
            print(f"    Winner accuracy : {w_avg:.1%}")
            print(f"    Top-10 Jaccard  : {jac_avg:.3f}")
            print(f"    Position MAE    : {mae_avg:.2f} places")
    else:
        metrics_df = pd.DataFrame()
        print("  (No completed 2026 race results in cache yet)")

    pred_df.to_csv(PRED_2026, index=False)
    print(f"\n  → Predictions saved: {PRED_2026}")

    # step 6: championship projection and plots
    print("\n[6/6] Projecting WDC standings and generating plots...")
    wdc_df = project_wdc(pred_df)
    wdc_df.to_csv(WDC_2026, index=False)
    print(f"  → WDC projection saved: {WDC_2026}")

    n_total = n_races_predicted
    try:
        schedule = fastf1.get_event_schedule(PREDICT_YEAR, include_testing=False)
        n_total  = len(schedule)
    except Exception:
        pass  # if schedule fetch fails just use what we have

    print_wdc_projection(wdc_df, n_races_predicted, n_total)
    txt_path = write_wdc_predictions_txt(wdc_df, n_races_predicted, n_total, DATA_DIR)
    plot_wdc_projection(wdc_df, n_races_predicted, PLOT_DIR)
    plot_season_predictions(pred_df, PLOT_DIR)
    plot_accuracy_timeline(metrics_df, PLOT_DIR)

    print("\n" + "=" * 60)
    print("  2026 Prediction Complete.")
    print(f"  Plots:       {PLOT_DIR}/")
    print(f"  Data:        {DATA_DIR}/")
    print(f"  Predictions: {PRED_2026}")
    print(f"  WDC table:   {WDC_2026}")
    print(f"  Rankings txt:{txt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
