"""
mach_learning.py — F1 Race Prediction: Random Forest + XGBoost
===============================================================
Trains and evaluates two complementary models to predict Formula 1 race outcomes:

  1. XGBoost Regressor   → predicts numeric finishing position (rank by predicted pos)
  2. Random Forest       → predicts winner (binary) and top-10 (binary)

APPROACH:
  - Training data: 2022–2024 races with pre-race features
    (FP pace, qualifying position, rolling form, circuit history)
  - Validation: Leave-one-season-out cross-validation
    (train on 2022+2023, test on 2024)
  - Final model: trained on all 2022–2024 data
  - Prediction: apply to 2025 pre-race features from feature_engineering.py

KEY DESIGN DECISION — position-based framing:
  We predict *finishing position* as a regression problem (XGBoost),
  then rank all 20 drivers per race to get the predicted order.
  The Random Forest handles binary tasks (win / top-10) as a sanity check.
  The ensemble combines both by ranking XGBoost positions and weighting
  RF probabilities.

HOW TO RUN (from project root):
    # Step 1: generate features
    python final_report/feature_engineering.py

    # Step 2: train and evaluate
    python final_report/mach_learning.py

OUTPUT:
    data_organized/model_evaluation.txt    — accuracy metrics + validation summary
    data_organized/feature_importances.csv — ranked feature importances
    data_organized/2025_predictions.csv    — predicted race results for 2025

REFERENCES:
    Random Forest: Breiman (2001), scikit-learn RandomForestClassifier/Regressor
    XGBoost:       Chen & Guestrin (2016), https://xgboost.readthedocs.io/
    F1 ML example: https://f1-predictor.readthedocs.io/
    FastF1 docs:   https://docs.fastf1.dev/
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, classification_report
)
from sklearn.impute import SimpleImputer

import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

DATA_DIR   = "data_organized"
TRAIN_FILE = os.path.join(DATA_DIR, "ml_training_features.csv")
PRED_FILE  = os.path.join(DATA_DIR, "ml_2025_features.csv")
OUT_DIR    = DATA_DIR

# Features used by the models (all numeric)
# These must match columns produced by feature_engineering.py
FEATURE_COLS = [
    "grid_position",
    "fp1_pace_delta",
    "fp2_pace_delta",
    "fp3_pace_delta",
    "sprint_quali_pos",
    "sprint_pos",
    "rolling_avg_finish_5",
    "rolling_dnf_rate_5",
    "rolling_points_5",
    "team_rolling_avg_5",
    "circuit_driver_avg_finish",
    "circuit_team_avg_finish",
    "IsSprintWeekend",
]

# Random seed for reproducibility
SEED = 42

# ─────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────

def load_and_prepare(filepath, fit_imputer=None):
    """
    Load the feature CSV and return (X, y_pos, y_win, y_top10, df_meta, imputer).

    - X          : feature matrix (NaN imputed with column median)
    - y_pos      : target — finish_position (regression)
    - y_win      : target — is_winner (binary classification)
    - y_top10    : target — is_top10 (binary classification)
    - df_meta    : identifier columns (Year, Round, Driver, Team, EventName)
    - imputer    : fitted SimpleImputer (reuse for prediction set)

    Imputation strategy:
      NaN in FP deltas → impute with median (driver did not participate in session)
      NaN in rolling stats → impute with median (new driver or first few races)
      NaN in grid_position → impute with 20 (back of grid assumption)

    Ref: https://scikit-learn.org/stable/modules/impute.html
    """
    df = pd.read_csv(filepath)

    meta_cols = ["Year", "Round", "EventName", "Circuit", "Driver", "Team"]
    target_cols = ["finish_position", "is_winner", "is_top10"]

    df_meta = df[meta_cols].copy()
    X_raw   = df[FEATURE_COLS].copy().astype(float)

    # Impute missing values
    if fit_imputer is None:
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(
            imputer.fit_transform(X_raw),
            columns=FEATURE_COLS
        )
    else:
        imputer = fit_imputer
        X = pd.DataFrame(
            imputer.transform(X_raw),
            columns=FEATURE_COLS
        )

    y_pos   = df["finish_position"].astype(float) if "finish_position" in df else None
    y_win   = df["is_winner"].astype(float)       if "is_winner" in df else None
    y_top10 = df["is_top10"].astype(float)        if "is_top10" in df else None

    return X, y_pos, y_win, y_top10, df_meta, df, imputer


# ─────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────

def rank_predictions_per_race(df_meta, y_pred_pos):
    """
    Given predicted finishing positions, rank the drivers within each race.
    Returns a DataFrame with columns [Year, Round, Driver, PredictedPosition].

    This converts the regressor's raw numeric output into an ordered 1–20 ranking.
    """
    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy()
    df["raw_pred"] = y_pred_pos
    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["raw_pred"]
          .rank(method="first", ascending=True)
          .astype(int)
    )
    return df.sort_values(["Year", "Round", "PredictedPosition"])


def winner_accuracy(df_meta, y_pred_pos, y_true_pos):
    """
    Compute the fraction of races where the model correctly predicted the winner
    (i.e., the driver ranked 1st by the model actually won the race).
    """
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = y_pred_pos
    df["true_pos"] = y_true_pos.values

    correct = 0
    total   = 0
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().all():
            continue
        pred_winner = group.loc[group["pred_pos"].idxmin(), "Driver"]
        true_winner_rows = group[group["true_pos"] == 1]
        if true_winner_rows.empty:
            continue
        true_winner = true_winner_rows.iloc[0]["Driver"]
        correct += int(pred_winner == true_winner)
        total   += 1

    return correct / total if total > 0 else 0.0, correct, total


def top10_set_accuracy(df_meta, y_pred_pos, y_true_pos, k=10):
    """
    For each race, compute the Jaccard similarity between the predicted top-k
    and the actual top-k set of drivers.
    Returns the mean Jaccard score across all races.
    """
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = y_pred_pos
    df["true_pos"] = y_true_pos.values

    jaccard_scores = []
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().any():
            continue
        pred_top = set(
            group.nsmallest(k, "pred_pos")["Driver"].tolist()
        )
        true_top = set(
            group[group["true_pos"] <= k]["Driver"].tolist()
        )
        if not pred_top and not true_top:
            continue
        intersection = len(pred_top & true_top)
        union        = len(pred_top | true_top)
        jaccard_scores.append(intersection / union if union > 0 else 0.0)

    return np.mean(jaccard_scores) if jaccard_scores else 0.0


def average_position_mae(df_meta, y_pred_pos, y_true_pos):
    """
    Mean Absolute Error of the predicted rank vs actual finishing position.
    """
    ranked_df = rank_predictions_per_race(df_meta, y_pred_pos)
    merged = ranked_df[["Year", "Round", "Driver", "PredictedPosition"]].copy()
    merged["TruePosition"] = y_true_pos.values
    merged = merged.dropna(subset=["TruePosition"])
    if merged.empty:
        return np.nan
    return mean_absolute_error(merged["TruePosition"], merged["PredictedPosition"])


# ─────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────

def train_xgboost_regressor(X_train, y_train):
    """
    Train an XGBoost regressor to predict numeric finishing position.

    Hyperparameters chosen as a baseline from F1 prediction literature:
    - n_estimators=300: enough trees for stable predictions
    - max_depth=5: moderate depth to capture interactions without overfitting
    - learning_rate=0.05: slow learning for better generalization
    - subsample=0.8: row subsampling (reduces overfitting)
    - colsample_bytree=0.8: feature subsampling

    Ref: https://xgboost.readthedocs.io/en/stable/parameter.html
    """
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_winner(X_train, y_win_train):
    """
    Train a Random Forest binary classifier to predict race winner.

    class_weight='balanced' is critical: only 1 in ~20 drivers wins,
    so the class is heavily imbalanced. Balanced weighting forces the model
    to attend to positive examples.

    Ref: https://scikit-learn.org/stable/modules/generated/
         sklearn.ensemble.RandomForestClassifier.html
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    # Drop NaN targets
    valid = y_win_train.dropna()
    model.fit(X_train.loc[valid.index], valid.astype(int))
    return model


def train_random_forest_top10(X_train, y_top10_train):
    """
    Train a Random Forest binary classifier to predict top-10 finish.
    Less imbalanced than winner prediction (~50% positive rate).
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    valid = y_top10_train.dropna()
    model.fit(X_train.loc[valid.index], valid.astype(int))
    return model


# ─────────────────────────────────────────────────────────────────
# ENSEMBLE PREDICTION
# ─────────────────────────────────────────────────────────────────

def ensemble_predict(xgb_raw_pos, rf_win_proba, df_meta):
    """
    Combine XGBoost position predictions with RF winner probabilities
    to produce a final race ranking.

    Method:
      final_score = xgb_rank_score + alpha * (1 - rf_win_proba)

    Where:
      - xgb_rank_score: XGBoost predicted position (lower = better)
      - rf_win_proba: RF probability of winning (higher = more likely winner)
      - alpha: blending weight (0.3 gives XGBoost 70% influence)

    The combined score is then re-ranked per race.

    This is inspired by ensemble approaches in Kaggle F1 prediction competitions.
    """
    alpha = 0.3
    combined = xgb_raw_pos + alpha * (1 - rf_win_proba)

    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy()
    df["ensemble_score"] = combined
    df["rf_win_proba"]   = rf_win_proba
    df["xgb_raw_pred"]   = xgb_raw_pos

    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["ensemble_score"]
          .rank(method="first", ascending=True)
          .astype(int)
    )
    return df.sort_values(["Year", "Round", "PredictedPosition"])


# ─────────────────────────────────────────────────────────────────
# VALIDATION: Leave-one-season-out
# ─────────────────────────────────────────────────────────────────

def leave_one_season_out_validation(df_train, X_train_all, y_pos, y_win, y_top10):
    """
    Performs leave-one-season-out cross-validation.

    For each year Y in the training set:
      - Train on all years except Y
      - Test on year Y
      - Report winner accuracy, top-10 Jaccard, and position MAE

    This simulates the real use case: training on past seasons, predicting
    the next unseen season.
    """
    print("\n" + "─" * 55)
    print("  Leave-One-Season-Out Validation")
    print("─" * 55)

    years = sorted(df_train["Year"].unique())
    results = []

    for holdout_year in years:
        train_mask = df_train["Year"] != holdout_year
        test_mask  = df_train["Year"] == holdout_year

        X_tr = X_train_all[train_mask]
        X_te = X_train_all[test_mask]

        # Fit imputer on train split only
        imp = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=FEATURE_COLS)
        X_te = pd.DataFrame(imp.transform(X_te), columns=FEATURE_COLS)

        y_pos_tr  = y_pos[train_mask]
        y_pos_te  = y_pos[test_mask]
        y_win_tr  = y_win[train_mask]
        y_top10_tr = y_top10[train_mask]

        meta_te = df_train[test_mask][
            ["Year", "Round", "EventName", "Driver", "Team"]
        ].reset_index(drop=True)

        # Train models
        xgb_model = train_xgboost_regressor(X_tr, y_pos_tr.fillna(20))
        rf_win     = train_random_forest_winner(X_tr, y_win_tr)
        rf_top10   = train_random_forest_top10(X_tr, y_top10_tr)

        # Predict
        xgb_preds = xgb_model.predict(X_te)
        rf_win_proba = rf_win.predict_proba(X_te)[:, 1]
        rf_top10_pred = rf_top10.predict(X_te)

        # Metrics
        w_acc, w_correct, w_total = winner_accuracy(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )
        jaccard = top10_set_accuracy(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )
        mae = average_position_mae(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )

        results.append({
            "holdout_year": holdout_year,
            "winner_accuracy": w_acc,
            "winners_correct": w_correct,
            "total_races":     w_total,
            "top10_jaccard":   jaccard,
            "position_mae":    mae,
        })

        print(f"\n  Holdout {holdout_year}:")
        print(f"    Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)")
        print(f"    Top-10 Jaccard  : {jaccard:.3f}")
        print(f"    Position MAE    : {mae:.2f} places")

    val_df = pd.DataFrame(results)
    print(f"\n  ── Averages ──")
    print(f"    Winner accuracy : {val_df['winner_accuracy'].mean():.1%}")
    print(f"    Top-10 Jaccard  : {val_df['top10_jaccard'].mean():.3f}")
    print(f"    Position MAE    : {val_df['position_mae'].mean():.2f} places")

    return val_df


# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────

def print_feature_importances(xgb_model, rf_win_model, feature_names):
    """
    Display and return ranked feature importances from both models.
    XGBoost uses gain-based importance; RF uses impurity-based importance.
    """
    xgb_imp = pd.Series(
        xgb_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    rf_imp = pd.Series(
        rf_win_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("\n  XGBoost Feature Importances (top 10):")
    for feat, imp in xgb_imp.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    print("\n  Random Forest Feature Importances (top 10):")
    for feat, imp in rf_imp.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    combined = pd.DataFrame({
        "xgb_importance": xgb_imp,
        "rf_importance":  rf_imp,
    })
    combined["avg_importance"] = combined.mean(axis=1)
    combined = combined.sort_values("avg_importance", ascending=False)

    path = os.path.join(OUT_DIR, "feature_importances.csv")
    combined.to_csv(path)
    print(f"\n  → Feature importances saved to {path}")

    return combined


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  F1 Race Prediction — Model Training & Evaluation")
    print("=" * 55)

    # ── 1. Load training data ──────────────────────────────────────
    if not os.path.exists(TRAIN_FILE):
        print(f"\n[!] Training file not found: {TRAIN_FILE}")
        print("    Run feature_engineering.py first.")
        return

    print(f"\nLoading training data from {TRAIN_FILE}...")
    X_all, y_pos, y_win, y_top10, df_meta_all, df_train, imputer = \
        load_and_prepare(TRAIN_FILE)

    print(f"  Loaded {len(df_train)} rows "
          f"({df_train['Year'].nunique()} seasons, "
          f"{df_train['Driver'].nunique()} unique drivers)")

    # Drop rows with NaN targets
    valid_mask = y_pos.notna() & y_win.notna() & y_top10.notna()
    X_all    = X_all[valid_mask].reset_index(drop=True)
    y_pos    = y_pos[valid_mask].reset_index(drop=True)
    y_win    = y_win[valid_mask].reset_index(drop=True)
    y_top10  = y_top10[valid_mask].reset_index(drop=True)
    df_train = df_train[valid_mask].reset_index(drop=True)

    print(f"  After dropping NaN targets: {len(df_train)} rows")

    # ── 2. Leave-one-season-out cross-validation ───────────────────
    val_results = leave_one_season_out_validation(
        df_train, X_all, y_pos, y_win, y_top10
    )

    # ── 3. Train final models on ALL training data ─────────────────
    print("\n" + "─" * 55)
    print("  Training Final Models on All 2022–2024 Data")
    print("─" * 55)

    X_final = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_all),
        columns=FEATURE_COLS
    )

    xgb_final  = train_xgboost_regressor(X_final, y_pos.fillna(20))
    rf_win_final   = train_random_forest_winner(X_final, y_win)
    rf_top10_final = train_random_forest_top10(X_final, y_top10)
    print("  ✓ XGBoost regressor trained")
    print("  ✓ Random Forest winner classifier trained")
    print("  ✓ Random Forest top-10 classifier trained")

    # ── 4. Feature importances ─────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Feature Importances")
    print("─" * 55)
    feat_imp = print_feature_importances(xgb_final, rf_win_final, FEATURE_COLS)

    # ── 5. In-sample sanity check (full training set) ──────────────
    xgb_train_preds = xgb_final.predict(X_final)
    rf_win_train_proba = rf_win_final.predict_proba(X_final)[:, 1]

    w_acc, w_correct, w_total = winner_accuracy(
        df_train[["Year", "Round", "EventName", "Driver", "Team"]].reset_index(drop=True),
        xgb_train_preds,
        y_pos
    )
    jaccard = top10_set_accuracy(
        df_train[["Year", "Round", "EventName", "Driver", "Team"]].reset_index(drop=True),
        xgb_train_preds,
        y_pos
    )
    mae = average_position_mae(
        df_train[["Year", "Round", "EventName", "Driver", "Team"]].reset_index(drop=True),
        xgb_train_preds,
        y_pos
    )
    print(f"\n  In-sample training metrics:")
    print(f"    Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)")
    print(f"    Top-10 Jaccard  : {jaccard:.3f}")
    print(f"    Position MAE    : {mae:.2f} places")

    # ── 6. Predict 2025 races ──────────────────────────────────────
    if not os.path.exists(PRED_FILE):
        print(f"\n[!] 2025 prediction file not found: {PRED_FILE}")
        print("    Skipping 2025 predictions.")
    else:
        print("\n" + "─" * 55)
        print("  Predicting 2025 Race Results")
        print("─" * 55)

        X_2025, y_pos_2025, y_win_2025, y_top10_2025, \
            df_meta_2025, df_2025, _ = load_and_prepare(PRED_FILE, fit_imputer=imputer)

        xgb_2025_preds      = xgb_final.predict(X_2025)
        rf_win_2025_proba   = rf_win_final.predict_proba(X_2025)[:, 1]
        rf_top10_2025_pred  = rf_top10_final.predict(X_2025)

        pred_df = ensemble_predict(
            xgb_2025_preds, rf_win_2025_proba,
            df_2025[["Year", "Round", "EventName", "Driver", "Team"]]
        )
        pred_df["rf_top10_pred"] = rf_top10_2025_pred

        # If 2025 race results exist (for completed races), add them
        if "finish_position" in df_2025.columns and df_2025["finish_position"].notna().any():
            pred_df["actual_position"] = df_2025["finish_position"].values

        out_path = os.path.join(OUT_DIR, "2025_predictions.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"\n  → 2025 predictions saved to {out_path}")

        # Pretty-print first few races
        rounds = pred_df["Round"].unique()[:3]
        for rnd in rounds:
            race_pred = pred_df[pred_df["Round"] == rnd].head(10)
            race_name = race_pred.iloc[0]["EventName"]
            print(f"\n  2025 R{rnd:02d} {race_name} — Predicted Top 10:")
            print(f"    {'Pos':>3}  {'Driver':<6}  {'Team':<30}  {'Win%':>5}")
            print(f"    {'───':>3}  {'──────':<6}  {'──────────────────────────────':<30}  {'─────':>5}")
            for _, row in race_pred.iterrows():
                print(f"    {int(row['PredictedPosition']):>3}  "
                      f"{row['Driver']:<6}  "
                      f"{row['Team'][:30]:<30}  "
                      f"{row['rf_win_proba']*100:>4.1f}%")

    # ── 7. Save evaluation summary ─────────────────────────────────
    eval_path = os.path.join(OUT_DIR, "model_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write("F1 Race Prediction — Model Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write("Leave-One-Season-Out Cross-Validation:\n")
        f.write(val_results.to_string(index=False))
        f.write(f"\n\nMean Winner Accuracy : {val_results['winner_accuracy'].mean():.1%}\n")
        f.write(f"Mean Top-10 Jaccard  : {val_results['top10_jaccard'].mean():.3f}\n")
        f.write(f"Mean Position MAE    : {val_results['position_mae'].mean():.2f} places\n")
    print(f"\n  → Evaluation summary saved to {eval_path}")

    print("\n" + "=" * 55)
    print("  Done! Models trained and evaluated.")
    print("=" * 55)
    print(f"\nKey outputs:")
    print(f"  {os.path.join(OUT_DIR, 'feature_importances.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'model_evaluation.txt')}")
    print(f"  {os.path.join(OUT_DIR, '2025_predictions.csv')}")


if __name__ == "__main__":
    main()
