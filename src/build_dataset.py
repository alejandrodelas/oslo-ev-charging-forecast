"""
Build the station-hour modelling table for Oslo EV charging demand.

Inputs (data/raw):
  - nobil_events_oslo_2026-03.csv.gz      real-time status events (NOBIL websocket), Oslo only
  - nobil_oslo_2022_connectors_flat.xlsx  static station / connector metadata (NOBIL)
  - traffic_oslo_2026-03_hourly.csv.gz    hourly vehicle counts, 141 Oslo traffic points (Statens vegvesen)

Output (data/processed):
  - station_hour.csv.gz  one row per station and hour, with target + features

Key idea: instead of counting "how many events said CHARGING" (the original team proxy),
we rebuild the real state of every charger over time and compute a TIME-WEIGHTED occupancy:
the share of the hour that the station's operational chargers spent charging.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oslo_ev.occupancy import split_segments, replay_states, hourly_occupancy, shift_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Oslo"
STEP = "5min"                    # resolution used to rebuild charger states
GAP_HOURS = 2                    # no events for > 2h  => ingestion outage, states are reset


# ---------------------------------------------------------------- 1. events
ev = pd.read_csv(RAW / "nobil_events_oslo_2026-03.csv.gz", parse_dates=["received_at"])
n_raw = len(ev)
ev = ev.drop_duplicates(subset=["received_at", "charger_id", "status"])  # exact duplicates
ev = ev.dropna(subset=["charger_id", "status"])
print(f"events: {n_raw:,} raw -> {len(ev):,} after removing duplicates")

# detect ingestion outages (the collector was down ~31h on 24-25 March)
ev = split_segments(ev, max_gap=f"{GAP_HOURS}h")
print("continuous collection segments:\n", ev.groupby("segment")["received_at"].agg(["min", "max"]))

# ---------------------------------------------------------------- 2. charger states -> hourly occupancy
states = replay_states(ev, step=STEP)
sh = hourly_occupancy(states, min_snapshots=6)          # at least 30 min observed in the hour

# event-count proxy used by the original team (share of CHARGING events in the station-hour)
ev["hour"] = ev.received_at.dt.floor("h")
proxy = (ev.groupby(["station_id", "hour"])
           .agg(events=("status", "size"), charging_events=("status", lambda s: (s == "CHARGING").sum()))
           .reset_index())
proxy["charging_event_ratio"] = proxy.charging_events / proxy.events
sh = sh.merge(proxy[["station_id", "hour", "events", "charging_event_ratio"]], on=["station_id", "hour"], how="left")
sh["events"] = sh.events.fillna(0)

# ---------------------------------------------------------------- 3. static station metadata
st = pd.read_excel(RAW / "nobil_oslo_2022_connectors_flat.xlsx")
st["kw"] = (st.connector_attr_charging_capacity.astype(str)
              .str.extract(r"([\d,\.]+)\s*kW")[0].str.replace(",", ".").astype(float))
st["is_dc"] = st.connector_attr_charging_capacity.astype(str).str.contains("DC")
meta = (st.groupby("international_id")
          .agg(lat=("lat", "first"), lon=("lon", "first"), zipcode=("zipcode", "first"),
               n_points=("number_charging_points", "first"), max_kw=("kw", "max"),
               mean_kw=("kw", "mean"), share_dc=("is_dc", "mean"),
               open_24h=("station_attr_open_24h", lambda s: (s == "Yes").mean()),
               parking_fee=("station_attr_parking_fee", lambda s: (s == "Yes").mean()))
          .reset_index().rename(columns={"international_id": "station_id"}))
zip_agg = meta.groupby("zipcode").agg(zip_stations=("station_id", "size"), zip_points=("n_points", "sum")).reset_index()
meta = meta.merge(zip_agg, on="zipcode", how="left")
# distance to Oslo city centre (Oslo S), km
meta["dist_centre_km"] = np.hypot((meta.lat - 59.9111) * 111, (meta.lon - 10.7528) * 111 * np.cos(np.radians(59.91)))
sh = sh.merge(meta, on="station_id", how="left")

# ---------------------------------------------------------------- 4. traffic
tr = pd.read_csv(RAW / "traffic_oslo_2026-03_hourly.csv.gz")
tr["from"] = pd.to_datetime(tr["from"], utc=True)
tr = tr[(tr.coverage >= 90) & tr.volume.notna() & (tr.volume >= 0)]
# keep only points with >= 90 % of March hours valid, so the city total is comparable hour to hour
ok = tr.groupby("trp_id").size()
ok = ok[ok >= 0.9 * ok.max()].index
tr = tr[tr.trp_id.isin(ok)]
city = tr.groupby("from").volume.mean().rename("traffic_now").to_frame()
city["local"] = city.index.tz_convert(TZ)
city["dow"], city["hod"] = city.local.dt.dayofweek, city.local.dt.hour
# "Master Traffic Profile": typical traffic for each weekday x hour, from the whole month
profile = city.groupby(["dow", "hod"]).traffic_now.mean().rename("traffic_profile").reset_index()
print(f"traffic: {len(ok)} points kept, {len(city)} hours")

sh["local"] = sh.hour.dt.tz_convert(TZ)
sh["dow"], sh["hod"] = sh.local.dt.dayofweek, sh.local.dt.hour
sh = sh.merge(profile, on=["dow", "hod"], how="left")
sh = sh.merge(city[["traffic_now"]], left_on="hour", right_index=True, how="left")

# ---------------------------------------------------------------- 5. time features, lags, targets
sh["hour_sin"], sh["hour_cos"] = np.sin(2 * np.pi * sh.hod / 24), np.cos(2 * np.pi * sh.hod / 24)
sh["is_weekend"] = (sh.dow >= 5).astype(int)
sh = sh.sort_values(["station_id", "hour"]).reset_index(drop=True)

g = sh.groupby(["station_id", "segment"])
shifted = lambda col, k: shift_exact(sh, col, k)

for k in (1, 2, 3):
    sh[f"lag_{k}"] = shifted("occupancy", -k)
sh["rolling_3"] = sh[["lag_1", "lag_2", "lag_3"]].mean(axis=1, skipna=False)
sh["station_mean_so_far"] = g.occupancy.transform(lambda s: s.shift(1).expanding().mean())
for h in (1, 3, 6):
    sh[f"target_t{h}"] = shifted("occupancy", h)
    sh[f"traffic_profile_t{h}"] = shifted("traffic_profile", h)

sh = sh.drop(columns=["local"])
sh.to_csv(OUT / "station_hour.csv.gz", index=False)
print(f"station-hour table: {len(sh):,} rows, {sh.station_id.nunique()} stations, "
      f"{sh.hour.nunique()} hours -> data/processed/station_hour.csv.gz")
