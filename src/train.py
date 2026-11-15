"""
Train and evaluate EV charging-occupancy forecasters for Oslo at t+1, t+3 and t+6 hours.

Split is strictly chronological: the model is trained on the past and tested on the
last ~17 hours it has never seen (no shuffling, no future information in the features).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
TEST_START = pd.Timestamp("2026-03-26 08:00", tz="UTC")
PEAK = 0.8

df = pd.read_csv(ROOT / "data/processed/station_hour.csv.gz", parse_dates=["hour"])
df["occ_now"] = df["occupancy"]

STATIC = ["operational_chargers", "n_points", "max_kw", "mean_kw", "share_dc", "open_24h",
          "parking_fee", "zip_stations", "zip_points", "dist_centre_km", "lat", "lon"]
DYNAMIC = ["occ_now", "lag_1", "lag_2", "lag_3", "rolling_3", "station_mean_so_far"]


def features(h):
    d = df.copy()
    th = (d.hod + h) % 24                       # hour of day we are predicting
    d["target_hour_sin"], d["target_hour_cos"] = np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24)
    cols = DYNAMIC + ["hour_sin", "hour_cos", "target_hour_sin", "target_hour_cos",
                      "traffic_now", f"traffic_profile_t{h}"] + STATIC
    d = d.dropna(subset=DYNAMIC + [f"target_t{h}", f"traffic_profile_t{h}"])
    return d, cols


def models():
    return {
        "Linear regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                               n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                           min_samples_leaf=40, random_state=SEED),
    }


def scores(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5, "R2": r2_score(y, p)}


results, store = [], {}
for h in (1, 3, 6):
    d, cols = features(h)
    tr, te = d[d.hour < TEST_START - pd.Timedelta(hours=h)], d[d.hour >= TEST_START]  # gap: no target overlap
    y_tr, y_te = tr[f"target_t{h}"], te[f"target_t{h}"]
    preds = {
        "Baseline: average": np.full(len(te), y_tr.mean()),
        "Baseline: same as now": te.occ_now.values,
    }
    fitted = {}
    for name, m in models().items():
        m.fit(tr[cols], y_tr)
        preds[name] = np.clip(m.predict(te[cols]), 0, 1)
        fitted[name] = m
    for name, p in preds.items():
        results.append({"horizon": f"t+{h}", "model": name, **scores(y_te, p),
                        "n_train": len(tr), "n_test": len(te)})
    store[h] = dict(te=te, y=y_te, preds=preds, fitted=fitted, cols=cols, tr=tr)
    keep = te[["station_id", "hour", "hod", "operational_chargers", "n_points", "dist_centre_km", f"target_t{h}"]]
    keep.assign(**{f"pred_{k.split(':')[-1].strip().lower().replace(' ', '_')}": v for k, v in preds.items()}) \
        .round(4).to_csv(ROOT / f"reports/test_predictions_t{h}.csv.gz", index=False)
    print(f"t+{h}: train {len(tr):,} rows ({tr.hour.min():%d %b %H}h-{tr.hour.max():%d %b %H}h UTC) | "
          f"test {len(te):,} rows ({te.hour.min():%d %b %H}h-{te.hour.max():%d %b %H}h UTC)")

res = pd.DataFrame(results)
print(res.round(4).to_string(index=False))
res.round(4).to_csv(ROOT / "reports/metrics.csv", index=False)

# ---------------------------------------------------------------- best model + peak alerts (t+1)
best = "Gradient Boosting"   # lowest error at every horizon (Random Forest a close second)
s1 = store[1]
yb, pb = (s1["y"] >= PEAK).astype(int), s1["preds"][best]
peak_rows = []
for thr in np.arange(0.5, 0.95, 0.05):
    a = (pb >= thr).astype(int)
    peak_rows.append({"threshold": round(thr, 2), "precision": precision_score(yb, a, zero_division=0),
                      "recall": recall_score(yb, a), "f1": f1_score(yb, a)})
peaks = pd.DataFrame(peak_rows)
peaks.round(4).to_csv(ROOT / "reports/peak_alerts.csv", index=False)
print(peaks.round(3).to_string(index=False))

# permutation importance on the test set (what the model really relies on)
imp = permutation_importance(s1["fitted"][best], s1["te"][s1["cols"]], s1["y"], n_repeats=5,
                             random_state=SEED, n_jobs=-1, scoring="neg_mean_absolute_error")
imp = pd.Series(imp.importances_mean, index=s1["cols"]).sort_values(ascending=False)
s6 = store[6]
imp6 = permutation_importance(s6["fitted"][best], s6["te"][s6["cols"]], s6["y"], n_repeats=5,
                              random_state=SEED, n_jobs=-1, scoring="neg_mean_absolute_error")
imp6 = pd.Series(imp6.importances_mean, index=s6["cols"]).sort_values(ascending=False)
pd.DataFrame({"t+1": imp, "t+6": imp6}).round(5).to_csv(ROOT / "reports/feature_importance.csv")

summary = {
    "rows": int(len(df)), "stations": int(df.station_id.nunique()), "hours": int(df.hour.nunique()),
    "test_start_utc": str(TEST_START), "peak_threshold": PEAK,
    "best_model": best,
    "t1": res[(res.horizon == "t+1")].set_index("model")[["MAE", "RMSE", "R2"]].round(4).to_dict("index"),
    "peak_at_0.8": peaks[peaks.threshold == 0.8].iloc[0].round(4).to_dict(),
    "top_features_t1": imp.head(8).round(5).to_dict(),
    "top_features_t6": imp6.head(8).round(5).to_dict(),
}
json.dump(summary, open(ROOT / "reports/summary.json", "w"), indent=2)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
order = ["Baseline: average", "Baseline: same as now", "Linear regression", "Random Forest", "Gradient Boosting"]
colors = ["#b8b8b8", "#8a8a8a", "#7fa7c9", "#5b8db8", "#d9822b"]

# 1. MAE by model and horizon
fig, ax = plt.subplots(figsize=(8, 4.4))
w = 0.16
for i, m in enumerate(order):
    v = [res[(res.model == m) & (res.horizon == f"t+{h}")].MAE.iloc[0] for h in (1, 3, 6)]
    ax.bar(np.arange(3) + (i - 2) * w, v, w, label=m, color=colors[i])
ax.set_xticks(range(3), ["1 hour ahead", "3 hours ahead", "6 hours ahead"])
ax.set_ylabel("MAE (occupancy, 0-1)  ·  lower is better")
ax.set_title("Forecast error by model and horizon (test set)")
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig(FIG / "mae_by_model_and_horizon.png"); plt.close(fig)

# 2. feature importance
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, ser, lab in ((axes[0], imp, "1 hour ahead"), (axes[1], imp6, "6 hours ahead")):
    top = ser.head(10)[::-1]
    ax.barh(top.index, top.values, color="#d9822b")
    ax.set_title(lab); ax.set_xlabel("Increase in MAE when shuffled")
fig.suptitle(f"What the {best} model relies on")
fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)

# 3. city-wide occupancy over the test window: actual vs predicted
te = s1["te"].assign(pred=pb)
city = te.groupby("hour").agg(actual=("target_t1", "mean"), predicted=("pred", "mean"))
city.index = (city.index + pd.Timedelta(hours=1)).tz_convert("Europe/Oslo")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(city.index, city.actual, color="#333", lw=2, label="Actual")
ax.plot(city.index, city.predicted, color="#d9822b", lw=2, ls="--", label=f"{best}, 1 h ahead")
ax.set_ylabel("Average occupancy, all stations"); ax.legend(frameon=False)
ax.set_title("Oslo-wide charging occupancy on the unseen test day (Oslo time)")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "test_day_actual_vs_predicted.png"); plt.close(fig)

# 4. daily profile (all data) with traffic
prof = df.groupby("hod").agg(occ=("occupancy", "mean"), traffic=("traffic_profile", "mean"))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(prof.index, prof.occ, color="#d9822b", lw=2, label="Charging occupancy")
ax.set_ylabel("Average occupancy"); ax.set_xlabel("Hour of day (Oslo time)")
ax2 = ax.twinx(); ax2.plot(prof.index, prof.traffic, color="#5b8db8", lw=1.5, ls=":", label="Road traffic")
ax2.set_ylabel("Vehicles / hour per sensor"); ax2.spines["top"].set_visible(False)
ax.set_title("Chargers fill up at night, when traffic is lowest")
fig.legend(frameon=False, loc="center", bbox_to_anchor=(0.47, 0.47), ncol=1, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "daily_profile_occupancy_vs_traffic.png"); plt.close(fig)

# 5. peak-alert threshold trade-off
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(peaks.threshold, peaks.precision, marker="o", color="#d9822b", label="Precision (alerts that were right)")
ax.plot(peaks.threshold, peaks.recall, marker="o", color="#5b8db8", label="Recall (peaks we caught)")
ax.axvline(0.8, color="#999", ls=":", lw=1)
ax.set_xlabel("Alert threshold on predicted occupancy"); ax.set_ylim(0, 1)
ax.set_title("Saturation alerts (occupancy ≥ 80 %), 1 h ahead"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "peak_alert_tradeoff.png"); plt.close(fig)
print("figures saved to reports/figures")
