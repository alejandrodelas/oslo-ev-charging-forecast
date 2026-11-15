"""
Faithful Python port of the original team model (R, `modelosproy3final.Rmd`, May 2026),
run on the same raw sources, so the two approaches can be compared side by side.

Original design, kept on purpose:
  * one row per station and hour *in which the station sent at least one message*
  * target `charging_ratio` = share of the station's messages in that hour that say CHARGING
  * features: lags of charging_ratio, hour, weekend, "master traffic profile" (weekday x hour),
    station size / capacity, share of AVAILABLE / BLOCKED / OUTOFORDER messages, number of
    messages, location and zipcode context
  * chronological 80/20 split, Random Forest with 300 trees, predictions clipped to [0, 1]
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"

ev = pd.read_csv(RAW / "nobil_events_oslo_2026-03.csv.gz", parse_dates=["received_at"])
nobil = pd.read_excel(RAW / "nobil_oslo_2022_connectors_flat.xlsx")
tr = pd.read_csv(RAW / "traffic_oslo_2026-03_hourly.csv.gz")

# ---- 1. traffic "master profile": mean volume per weekday (Mon=1) and hour
tr["from"] = pd.to_datetime(tr["from"], utc=True)
tr["wday"], tr["hour"] = tr["from"].dt.dayofweek + 1, tr["from"].dt.hour
profile = tr.groupby(["wday", "hour"]).volume.mean().rename("traffic_volume_avg").reset_index()

# ---- 2. station metadata + zipcode context (as in the R code)
meta = (nobil.groupby("international_id")
             .agg(zipcode=("zipcode", "first"), lat=("lat", "first"), lon=("lon", "first"),
                  number_charging_points=("number_charging_points", "max"),
                  capacity_kw=("connector_attr_charging_capacity", "first")).reset_index()
             .rename(columns={"international_id": "station_id"}))
# R: as.numeric(gsub("[^0-9.]", "", capacity)) -- note "62,5 kW DC" becomes 625 (decimal comma lost)
meta["capacity_kw_num"] = pd.to_numeric(meta.capacity_kw.astype(str).str.replace(r"[^0-9.]", "", regex=True),
                                        errors="coerce")
z = meta.groupby("zipcode").agg(stations_in_zipcode=("station_id", "nunique"),
                                charging_points_in_zipcode=("number_charging_points", "sum")).reset_index()
meta = meta.merge(z, on="zipcode")

# ---- 3. hourly aggregation of events
ev["hour_ts"] = ev.received_at.dt.floor("h")
for s in ("CHARGING", "AVAILABLE", "BLOCKED", "OUTOFORDER"):
    ev[s] = ev.status.eq(s)
hourly = (ev.groupby(["station_id", "hour_ts"])
            .agg(n_events=("status", "size"), charging_ratio=("CHARGING", "mean"),
                 available_ratio=("AVAILABLE", "mean"), blocked_ratio=("BLOCKED", "mean"),
                 outoforder_ratio=("OUTOFORDER", "mean")).reset_index())
hourly["hour"], hourly["wday"] = hourly.hour_ts.dt.hour, hourly.hour_ts.dt.dayofweek + 1
df = (hourly.merge(meta, on="station_id", how="left").merge(profile, on=["wday", "hour"], how="left")
            .sort_values(["station_id", "hour_ts"]).reset_index(drop=True))
df["is_weekend"] = (df.wday >= 6).astype(int)
g = df.groupby("station_id")
gap = lambda k: (g.hour_ts.shift(-k) - df.hour_ts).dt.total_seconds() / 3600
df["gap_prev"] = (df.hour_ts - g.hour_ts.shift(1)).dt.total_seconds() / 3600
for k in (1, 2, 3):
    df[f"lag_{k}"] = g.charging_ratio.shift(k)          # row lags, as lag() in dplyr
for h in (1, 3, 6):
    df[f"gap_next_{h}"] = gap(h)
    df[f"target_t{h}"] = g.charging_ratio.shift(-h)     # the h-th next *row*, kept only if it is h hours later

X = ["lag_1", "lag_2", "lag_3", "hour", "is_weekend", "traffic_volume_avg", "number_charging_points",
     "capacity_kw_num", "n_events", "available_ratio", "blocked_ratio", "outoforder_ratio",
     "lat", "lon", "stations_in_zipcode", "charging_points_in_zipcode"]

rows, out = [], {}
for h in (1, 3, 6):
    m = df[(df.gap_prev == 1) & (df[f"gap_next_{h}"] == h)].dropna(subset=X + [f"target_t{h}"])
    m = m.sort_values(["hour_ts", "station_id"])
    cut = int(0.8 * len(m))
    trn, tst = m.iloc[:cut], m.iloc[cut:]
    y, yt = trn[f"target_t{h}"], tst[f"target_t{h}"]
    rf = RandomForestRegressor(n_estimators=300, max_features=1 / 3, n_jobs=-1, random_state=123)
    p = np.clip(rf.fit(trn[X], y).predict(tst[X]), 0, 1)
    preds = {"Baseline: average": np.full(len(tst), y.mean()), "Baseline: lag_1": tst.lag_1.values,
             "Random Forest (team)": p}
    for name, pr in preds.items():
        rows.append({"horizon": f"t+{h}", "model": name, "MAE": mean_absolute_error(yt, pr),
                     "RMSE": mean_squared_error(yt, pr) ** 0.5, "R2": r2_score(yt, pr),
                     "n_rows": len(m), "n_stations": m.station_id.nunique()})
    if h == 1:
        a, b = (yt >= 0.8).astype(int), (p >= 0.8).astype(int)
        out["peak_precision_0.8"], out["peak_recall_0.8"] = precision_score(a, b), recall_score(a, b)
        # how much does the event ratio say about real occupancy? (same station-hours)
        occ = pd.read_csv(ROOT / "data/processed/station_hour.csv.gz", parse_dates=["hour"])
        j = df.merge(occ[["station_id", "hour", "occupancy"]], left_on=["station_id", "hour_ts"],
                     right_on=["station_id", "hour"], how="inner")
        out["corr_event_ratio_vs_true_occupancy"] = j[["charging_ratio", "occupancy"]].corr().iloc[0, 1]
        out["share_station_hours_with_no_message"] = 1 - len(hourly) / max(1, len(occ))

res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
print({k: round(v, 4) for k, v in out.items()})
res.round(4).to_csv(ROOT / "reports/team_replication_metrics.csv", index=False)
json.dump({k: round(float(v), 4) for k, v in out.items()}, open(ROOT / "reports/team_replication_extra.json", "w"), indent=2)
