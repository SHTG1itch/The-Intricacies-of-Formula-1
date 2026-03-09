# Plots each F1 constructor's final championship points across seasons (2018-2026).
# Single plot: x = year, y = points. Team rebrands are merged into one line.

import os
import logging
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CACHE_DIR = "discovery_raw_data"
OUTPUT_DIR = "team_performance_plots"
YEARS = list(range(2018, 2027))

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Map every historical name to a single lineage label
TEAM_LINEAGE = {
    # Mercedes
    "Mercedes":              "Mercedes",
    "Mercedes-AMG Petronas F1 Team": "Mercedes",
    # Red Bull
    "Red Bull Racing":       "Red Bull Racing",
    "Red Bull":              "Red Bull Racing",
    "Oracle Red Bull Racing":"Red Bull Racing",
    # Ferrari
    "Ferrari":               "Ferrari",
    "Scuderia Ferrari":      "Ferrari",
    # McLaren
    "McLaren":               "McLaren",
    "McLaren F1 Team":       "McLaren",
    # Renault -> Alpine
    "Renault":               "Alpine (Renault)",
    "Renault F1 Team":       "Alpine (Renault)",
    "Alpine":                "Alpine (Renault)",
    "Alpine F1 Team":        "Alpine (Renault)",
    "BWT Alpine F1 Team":    "Alpine (Renault)",
    # Force India -> Racing Point -> Aston Martin
    "Force India":           "Aston Martin (Force India)",
    "Racing Point":          "Aston Martin (Force India)",
    "Aston Martin":          "Aston Martin (Force India)",
    "Aston Martin Aramco Mercedes": "Aston Martin (Force India)",
    # Toro Rosso -> AlphaTauri -> RB
    "Toro Rosso":            "RB (Toro Rosso)",
    "Scuderia Toro Rosso":   "RB (Toro Rosso)",
    "AlphaTauri":            "RB (Toro Rosso)",
    "Scuderia AlphaTauri":   "RB (Toro Rosso)",
    "RB":                    "RB (Toro Rosso)",
    "Visa Cash App RB":      "RB (Toro Rosso)",
    "Racing Bulls":          "RB (Toro Rosso)",
    "Visa Cash App RB F1 Team": "RB (Toro Rosso)",
    # Sauber -> Alfa Romeo -> Kick Sauber
    "Sauber":                "Sauber (Alfa Romeo)",
    "Alfa Romeo":            "Sauber (Alfa Romeo)",
    "Alfa Romeo Racing":     "Sauber (Alfa Romeo)",
    "Kick Sauber":           "Sauber (Alfa Romeo)",
    "Stake F1 Team Kick Sauber": "Sauber (Alfa Romeo)",
    # Williams
    "Williams":              "Williams",
    "Williams Racing":       "Williams",
    # Haas
    "Haas F1 Team":          "Haas",
    "MoneyGram Haas F1 Team":"Haas",
    "Haas":                  "Haas",
}

TEAM_COLORS = {
    "Mercedes":                     "#00D2BE",
    "Red Bull Racing":              "#1E41FF",
    "Ferrari":                      "#DC0000",
    "McLaren":                      "#FF8000",
    "Alpine (Renault)":             "#0090FF",
    "Aston Martin (Force India)":   "#006F62",
    "RB (Toro Rosso)":              "#6692FF",
    "Sauber (Alfa Romeo)":          "#900000",
    "Williams":                     "#005AFF",
    "Haas":                         "#B6BABD",
}


def canonical(name):
    name = name.strip()
    if name in TEAM_LINEAGE:
        return TEAM_LINEAGE[name]
    # Fuzzy match
    low = name.lower()
    for key, val in TEAM_LINEAGE.items():
        if key.lower() in low or low in key.lower():
            return val
    return name


def race_points(position, fastest_lap=False, year=2019):
    pts = POINTS_TABLE.get(int(position), 0) if pd.notna(position) else 0
    if fastest_lap and year >= 2019 and int(position) <= 10:
        pts += 1
    return pts


def season_totals(year):
    """Returns {canonical_team: total_pts} for the year."""
    print(f"  {year}...", end=" ", flush=True)
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"[ERR] {e}")
        return {}

    team_pts = {}
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        try:
            s = fastf1.get_session(year, rnd, "Race")
            s.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception:
            continue

        res = s.results
        if res is None or res.empty:
            continue

        cols = {c.lower(): c for c in res.columns}
        team_col = cols.get("teamname", cols.get("constructorname"))
        pos_col  = cols.get("position", cols.get("classifiedposition"))
        fl_col   = cols.get("fastestlap")
        if not team_col or not pos_col:
            continue

        for _, row in res.iterrows():
            team = canonical(str(row[team_col]))
            if not team or team.lower() in ("nan", "none", ""):
                continue
            try:
                pos = int(float(row[pos_col]))
            except (ValueError, TypeError):
                pos = 99
            has_fl = bool(row[fl_col]) if fl_col and pd.notna(row.get(fl_col)) else False
            team_pts[team] = team_pts.get(team, 0) + race_points(pos, has_fl, year)

    print(f"done ({len(team_pts)} teams)")
    return team_pts


def main():
    # {team: {year: pts}}
    data = {}
    for year in YEARS:
        for team, pts in season_totals(year).items():
            data.setdefault(team, {})[year] = pts

    if not data:
        print("No data - run cache_fastf1_data.py first.")
        return

    # Only keep teams with 2+ seasons of data
    teams = sorted(t for t, d in data.items() if len(d) >= 2)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    for team in teams:
        d = data[team]
        xs = [y for y in YEARS if y in d]
        ys = [d[y] for y in YEARS if y in d]
        color = TEAM_COLORS.get(team, "#888888")

        ax.plot(xs, ys, label=team, color=color,
                linewidth=2.5, marker="o", markersize=6, alpha=0.92)

        # Label at the last data point
        if xs:
            ax.annotate(team, xy=(xs[-1], ys[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=color, va="center")

    ax.set_xlabel("Season", fontsize=13, color="#cccccc")
    ax.set_ylabel("Championship Points", fontsize=13, color="#cccccc")
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=10, color="#aaaaaa")
    ax.tick_params(axis="y", colors="#aaaaaa")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid(axis="y", color="#2a2a4a", linewidth=0.5)
    ax.grid(axis="x", color="#2a2a4a", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.legend(fontsize=8, loc="upper left", framealpha=0.25,
              facecolor="#0d0d1a", edgecolor="#333355", labelcolor="white", ncol=2)

    fig.suptitle("F1 Constructor Championship Points Over Time",
                 fontsize=17, fontweight="bold", color="white", y=0.97)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "team_performance_over_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    print("=== F1 Team Performance Over Time ===")
    main()
    print("Done.")
