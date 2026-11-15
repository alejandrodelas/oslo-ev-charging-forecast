# Project history

How the original university project ran (Feb – May 2026), and how this repository relates to it.

## The original team project

*Proyecto III*, BSc Data Science, Universitat Politècnica de València. Project code **DS-PRO-III.16, "EVINY-PREDICTING"**: a 6-person team working on a case proposed for the Norwegian energy company Eviny.

| When | Milestone | What happened |
|---|---|---|
| Feb 2026 | Initiation | Problem definition, objectives, roles, project plan (Gantt chart in TeamGantt), tasks tracked in Notion |
| 9 Mar | **M1 presentation** | Objectives, data sources (NOBIL, Statens vegvesen traffic, MET Norway weather), roles, plan, costs, benefits and risks |
| Mar – Apr | Data collection | NOBIL has no history of charger status, so the team built a real-time collector (websocket → JSONL → GitHub) on a university virtual machine. The machine stopped several times, which left only a few usable days |
| 20 Apr | **M2 report** | Data preparation, exploratory analysis, interactive station map, first models. Feedback: the model predicted poorly and needed improving or explaining |
| 7 May | Final model | R notebook: station-hour "minable view", linear regression for drivers, Random Forest / Gradient Boosting for t+1, t+3, t+6 |
| 11 May | **Final presentation** + poster | 15-minute talk; A0 poster for the ETSINF project fair |
| 15 May | **Final report** | 23-page report |
| 27–28 May | Project fair | Stand at the ETSINF projects fair |

### My part in the team project
Data side: preparing NOBIL's static station data (Oslo, stations from 2022 on), integrating one full day of real-time events with the station metadata, the exploratory analysis and the interactive station map (`notebooks/01_original_team_eda.ipynb`, `reports/oslo_stations_map.html`). I also presented the data part with a teammate, the costs, benefits and risks in M1, and worked on the poster.

Teammates built the real-time collector ([cofrian/nobil_data](https://github.com/cofrian/nobil_data)), the traffic and weather extraction, and the final R models.

## Problems the team ran into (and what this repo does about them)

| Problem in the original project | Effect | In this repository |
|---|---|---|
| The collector on the university VM stopped several times | Only ~3 usable days; a 31-hour hole on 24–25 March | Outages are detected and states, lags and targets never cross them |
| Target = share of status **messages** saying CHARGING | Only hours with a message can be modelled (about 3 in 4 station-hours are lost), and it correlates only r = 0.19 with real occupancy | Each charger's state is replayed every 5 minutes → **time-weighted real occupancy** for every station-hour |
| Different horizons evaluated on different station subsets (592 → 107 stations) | Error *seemed* to fall from t+1 to t+6 | All stations are kept at every horizon (52,000–69,000 station-hours); error rises with the horizon, as it should |
| Capacity text such as "62,5 kW DC" parsed to 625 | Wrong kW values for some stations | Decimal comma handled |
| Weather added complexity with little signal | Dropped in the final model | Not used (consistent with the original finding) |

The original model is reproduced in Python in [`src/replicate_team.py`](../src/replicate_team.py). It gets t+1 MAE 0.20 and R² 0.29, against the 0.19 and 0.28 in the final report.
