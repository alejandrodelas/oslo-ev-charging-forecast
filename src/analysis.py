"""
Extra analysis for the README:
  1. why the target was redefined (message ratio vs real occupancy)
  2. where the stations are and how busy they are (static map)
  3. where the model fails (error by station size and by hour of day)
  4. team model vs this version, side by side
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
ORANGE, BLUE, GREY = "#d9822b", "#5b8db8", "#9a9a9a"

sh = pd.read_csv(ROOT / "data/processed/station_hour.csv.gz", parse_dates=["hour"])

# 1. message ratio vs real occupancy ------------------------------------------------------------
both = sh.dropna(subset=["charging_event_ratio"])
r = both[["charging_event_ratio", "occupancy"]].corr().iloc[0, 1]
fig, ax = plt.subplots(figsize=(5.6, 4.6))
hb = ax.hexbin(both.charging_event_ratio, both.occupancy, gridsize=22, cmap="Oranges", mincnt=1, bins="log")
ax.set_xlabel("Team target: share of messages saying CHARGING")
ax.set_ylabel("Real occupancy (time-weighted)")
ax.set_title(f"The original target barely tracks real occupancy (r = {r:.2f})")
fig.colorbar(hb, ax=ax, label="station-hours (log)")
fig.tight_layout(); fig.savefig(FIG / "event_ratio_vs_true_occupancy.png"); plt.close(fig)
share_silent = 1 - len(both) / len(sh)

# 2. static map ------------------------------------------------------------------------------
st = sh.groupby("station_id").agg(lat=("lat", "first"), lon=("lon", "first"), occ=("occupancy", "mean"),
                                  size=("operational_chargers", "mean")).dropna()
fig, ax = plt.subplots(figsize=(6.4, 5.6))
sc = ax.scatter(st.lon, st.lat, c=st.occ, s=6 + 4 * np.sqrt(st["size"]), cmap="YlOrRd", vmin=0, vmax=1,
                alpha=0.8, linewidths=0)
ax.scatter([10.7528], [59.9111], marker="*", s=160, color="#222", label="Oslo S (centre)")
ax.set_xlim(10.6, 10.98); ax.set_ylim(59.81, 59.99)
ax.set_aspect(1 / np.cos(np.radians(59.91)))
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(frameon=False, loc="lower right")
ax.set_title("Oslo charging stations by average occupancy", fontsize=10)
fig.colorbar(sc, ax=ax, label="average occupancy", shrink=0.8)
fig.tight_layout(); fig.savefig(FIG / "stations_map.png"); plt.close(fig)

# 3. where does the model fail? -----------------------------------------------------------------
p6 = pd.read_csv(ROOT / "reports/test_predictions_t6.csv.gz")
p6["size"] = pd.cut(p6.operational_chargers, [0, 1.5, 2.5, 5.5, 100], labels=["1", "2", "3-5", "6+"])
err = lambda d, c: (d[c] - d.target_t6).abs().mean()
by_size = p6.groupby("size", observed=True).apply(
    lambda d: pd.Series({"same_as_now": err(d, "pred_same_as_now"), "gradient_boosting": err(d, "pred_gradient_boosting"),
                         "n": len(d)}), include_groups=False)
by_size.round(4).to_csv(ROOT / "reports/error_by_station_size_t6.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(len(by_size))
ax.bar(x - 0.2, by_size.same_as_now, 0.4, color=GREY, label="Baseline: same as now")
ax.bar(x + 0.2, by_size.gradient_boosting, 0.4, color=ORANGE, label="Gradient Boosting")
ax.set_xticks(x, [f"{i} charger{'' if i == '1' else 's'}\n(n={int(n):,})" for i, n in zip(by_size.index, by_size.n)])
ax.set_ylabel("MAE, 6 h ahead"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Small stations are the hardest: one car changes everything")
fig.tight_layout(); fig.savefig(FIG / "error_by_station_size.png"); plt.close(fig)

# 4. team model vs this version ---------------------------------------------------------------
team = pd.read_csv(ROOT / "reports/team_replication_metrics.csv")
mine = pd.read_csv(ROOT / "reports/metrics.csv")
extra = json.load(open(ROOT / "reports/team_replication_extra.json"))
cmp_rows = []
for h in ("t+1", "t+3", "t+6"):
    t = team[(team.horizon == h) & (team.model == "Random Forest (team)")].iloc[0]
    tb = team[(team.horizon == h) & (team.model == "Baseline: average")].iloc[0]
    m = mine[(mine.horizon == h) & (mine.model == "Gradient Boosting")].iloc[0]
    mb = mine[(mine.horizon == h) & (mine.model == "Baseline: average")].iloc[0]
    cmp_rows.append({"horizon": h, "team_rows": int(t.n_rows), "team_stations": int(t.n_stations),
                     "team_R2": t.R2, "team_MAE_vs_avg_baseline_pct": 100 * (1 - t.MAE / tb.MAE),
                     "this_rows": int(m.n_test + m.n_train), "this_R2": m.R2,
                     "this_MAE_vs_avg_baseline_pct": 100 * (1 - m.MAE / mb.MAE)})
cmp = pd.DataFrame(cmp_rows)
cmp.round(3).to_csv(ROOT / "reports/team_vs_this_version.csv", index=False)
print(cmp.round(3).to_string(index=False))
print(f"corr(message ratio, occupancy) = {r:.3f}; station-hours with no message = {share_silent:.1%}")
print(by_size.round(3))
