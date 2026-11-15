"""
Core logic: from a stream of charger status messages to time-weighted station occupancy.

NOBIL only sends a message when a charger changes state. To know how busy a station was
during an hour we therefore have to *replay* each charger's state over time:

    20:10 CHARGING ........................ 21:35 AVAILABLE
    |<-------------- charging -------------->|

and then measure what share of the hour its working chargers spent charging.
"""
from __future__ import annotations

import pandas as pd

OPERATIONAL = frozenset({"AVAILABLE", "CHARGING", "BLOCKED"})  # OUTOFORDER / UNKNOWN = not usable


def split_segments(events: pd.DataFrame, max_gap: str = "2h") -> pd.DataFrame:
    """Label continuous collection periods. A silence longer than `max_gap` across the whole
    network means the collector was down, so states must not be carried across it."""
    ev = events.sort_values("received_at").copy()
    ev["segment"] = (ev["received_at"].diff() > pd.Timedelta(max_gap)).cumsum()
    return ev


def replay_states(events: pd.DataFrame, step: str = "5min") -> pd.DataFrame:
    """Last known status of every charger on a regular time grid (forward fill, never across segments).

    events: columns received_at (tz-aware), charger_id, station_id, status, segment
    returns: long table ts, charger_id, station_id, status, segment
    """
    frames = []
    for seg, g in events.groupby("segment"):
        grid = pd.date_range(g.received_at.min().ceil(step), g.received_at.max().floor(step), freq=step)
        if len(grid) == 0:
            continue
        wide = g.pivot_table(index="received_at", columns="charger_id", values="status", aggfunc="last")
        wide = wide.sort_index().ffill().reindex(grid, method="ffill")
        long = wide.stack().rename("status").reset_index()
        long.columns = ["ts", "charger_id", "status"]
        long["segment"] = seg
        frames.append(long)
    states = pd.concat(frames, ignore_index=True)
    charger2station = events.drop_duplicates("charger_id").set_index("charger_id")["station_id"]
    states["station_id"] = states["charger_id"].map(charger2station)
    return states


def hourly_occupancy(states: pd.DataFrame, min_snapshots: int = 6) -> pd.DataFrame:
    """Time-weighted occupancy per station and hour = mean over the hour of
    (chargers CHARGING / chargers operational). Hours observed for less than
    `min_snapshots` grid points (30 min at 5-min steps) are dropped."""
    s = states.assign(operational=states["status"].isin(OPERATIONAL),
                      charging=states["status"].eq("CHARGING"),
                      hour=states["ts"].dt.floor("h"))
    snap = (s.groupby(["station_id", "segment", "hour", "ts"])
             .agg(op=("operational", "sum"), ch=("charging", "sum")).reset_index())
    snap = snap[snap["op"] > 0]
    snap["occ"] = snap["ch"] / snap["op"]
    out = (snap.groupby(["station_id", "segment", "hour"])
               .agg(occupancy=("occ", "mean"), operational_chargers=("op", "mean"),
                    n_snapshots=("occ", "size")).reset_index())
    return out[out["n_snapshots"] >= min_snapshots].reset_index(drop=True)


def shift_exact(df: pd.DataFrame, col: str, k: int, keys=("station_id", "segment"), time_col="hour") -> pd.Series:
    """Value of `col` exactly k hours later (k>0) or earlier (k<0) for the same station.
    Returns NaN when that exact hour is missing, so lags/targets never jump over gaps."""
    g = df.groupby(list(keys))
    val, when = g[col].shift(-k), g[time_col].shift(-k)
    return val.where(when == df[time_col] + pd.Timedelta(hours=k))
