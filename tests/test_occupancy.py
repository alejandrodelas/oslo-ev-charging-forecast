"""Unit tests for the occupancy logic on tiny hand-made examples."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oslo_ev.occupancy import split_segments, replay_states, hourly_occupancy, shift_exact  # noqa: E402

T = lambda s: pd.Timestamp(f"2026-03-23 {s}", tz="UTC")


def events(rows):
    return pd.DataFrame(rows, columns=["received_at", "station_id", "charger_id", "status"])


def test_state_is_carried_forward_until_next_message():
    ev = split_segments(events([
        (T("10:00"), "S1", "c1", "CHARGING"),
        (T("10:00"), "S1", "c2", "AVAILABLE"),
        (T("10:30"), "S1", "c1", "AVAILABLE"),
        (T("11:00"), "S1", "c2", "AVAILABLE"),   # keeps the grid running to 11:00
    ]))
    occ = hourly_occupancy(replay_states(ev, "5min"), min_snapshots=1)
    row = occ[occ.hour == T("10:00")].iloc[0]
    # c1 charges 10:00-10:25 (6 of 12 snapshots), c2 never -> mean occupancy = 0.5 * 6/12 = 0.25
    assert row.occupancy == pytest.approx(0.25)
    assert row.operational_chargers == 2


def test_out_of_order_chargers_are_not_counted():
    ev = split_segments(events([
        (T("10:00"), "S1", "c1", "CHARGING"),
        (T("10:00"), "S1", "c2", "OUTOFORDER"),
        (T("10:55"), "S1", "c1", "CHARGING"),
    ]))
    occ = hourly_occupancy(replay_states(ev, "5min"), min_snapshots=1)
    assert occ.occupancy.iloc[0] == pytest.approx(1.0)        # 1 working charger, always charging
    assert occ.operational_chargers.iloc[0] == 1


def test_states_are_not_carried_across_an_outage():
    ev = split_segments(events([
        (T("08:00"), "S1", "c1", "CHARGING"),
        (T("08:30"), "S1", "c1", "CHARGING"),
        (T("14:00"), "S1", "c1", "AVAILABLE"),   # 5.5 h of silence -> new segment
        (T("14:30"), "S1", "c1", "AVAILABLE"),
    ]), max_gap="2h")
    assert ev.segment.nunique() == 2
    occ = hourly_occupancy(replay_states(ev, "5min"), min_snapshots=1)
    assert set(occ.hour) == {T("08:00"), T("14:00")}          # nothing invented for 09:00-13:00


def test_shift_exact_returns_nan_when_hour_is_missing():
    df = pd.DataFrame({"station_id": ["S1"] * 3, "segment": [0] * 3,
                       "hour": [T("10:00"), T("11:00"), T("13:00")], "occupancy": [0.1, 0.2, 0.4]})
    nxt = shift_exact(df, "occupancy", 1)
    assert nxt.iloc[0] == 0.2                  # 10:00 -> 11:00 exists
    assert pd.isna(nxt.iloc[1])                # 11:00 -> 12:00 missing: no silent jump to 13:00
    prev = shift_exact(df, "occupancy", -1)
    assert pd.isna(prev.iloc[0]) and prev.iloc[1] == 0.1 and pd.isna(prev.iloc[2])
