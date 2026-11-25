# JalMarg

**Vehicle-aware routing around waterlogged roads, with time-to-passable forecasting from video.**

> *"Google Maps tells you there's traffic. It doesn't tell you there's 40 cm of water, or that the underpass will be safe for your scooter in 18 minutes."*

OOAD + Deep Learning project · Deepanshu Semwal · B.Tech CSE, 2nd year · UPES Dehradun

> **Confidential:** keep this repository private until the UPES IPR cell has been consulted about a provisional patent.

---

## 1. Problem

Every monsoon, roads and underpasses in Indian cities fill with water. Navigation apps don't know how deep the water is, whether *your* vehicle can get through, or when it will drain. Commuters guess, and scooters stall, cars flood and people get stranded.

JalMarg uses video to **measure water depth** on each road segment, **forecast how fast it will drain**, and work out a **time-to-passable for each vehicle class**. A router then answers: *"Wait 18 min, or detour +11 min?"*

### What's new
- Existing research estimates depth from a single image, or forecasts a rough danger class.
- Existing apps collect yes/no crowd reports.
- **No existing system turns video into vehicle-specific passability forecasts and uses them in time-dependent routing.**

---

## 2. Scope (10-day build)

| In scope | Out of scope (future work) |
|---|---|
| One small area of Dehradun (OSMnx), 4–5 segments that flood | City-wide live deployment |
| Recorded video replayed as a "live" stream | Real municipal CCTV integration |
| Still or slow-moving water; 6 vehicle classes | Fast currents, open manholes |
| Streamlit + Folium demo, FastAPI facade | Native mobile apps, push notifications |
| SQLite / in-memory storage behind repositories | PostgreSQL + PostGIS, Redis |
| Synthetic depth time series (generated numerically) | Blender-rendered synthetic scenes |

---

## 3. Pipeline

```
Inputs              Perception (per frame)          Temporal (per segment)         Decisions
──────              ──────────────────────          ──────────────────────         ─────────
Video / photo  ──►  Water + object segmentation ──► Kalman tracker (h, dh/dt) ──►  Time-to-passable per vehicle
                    (YOLO11-seg)                          │                               │
                         │                                ▼                               ▼
                    Depth per reference object      Recession forecaster          Time-dependent router
                    (wheel, person, kerb)           (physics prior + GRU,         ("wait N min" vs "detour")
                         │                           quantile outputs)                    │
                         ▼                                ▲                               ▼
                    Uncertainty-weighted fusion     Rainfall (Open-Meteo)          Streamlit map + advice
                    h ± σ
```

1. **Ingest:** a camera adapter reads frames (video file / RTSP / photo) and tags them with a segment and timestamp.
2. **Perceive:** segment the water and detect reference objects; fit an ellipse to each wheel.
3. **Measure:** each reference gives `h_r = f_r · L_r` with error σ_r; fuse them by inverse-variance weighting into `h ± σ`.
4. **Track:** a 1-D Kalman filter per segment smooths observations from all sources.
5. **Forecast:** a physics prior `dh/dt = α·R(t) − k·(h − h∞)` plus a GRU correction outputs depth quantiles (10/50/90%).
6. **Decide:** `t*_v = min{ t ≥ now : q₀.₉(h(t)) < d_v − m }`; a time-dependent Dijkstra/A* compares detour vs. wait-then-go.

---

## 4. Deep learning

### 4.1 Perception
- **Model:** YOLO11-seg (Ultralytics), fine-tuned for the `water` class on the Roadway Flooding + Flood Area Segmentation datasets.
- **Vehicles / people:** COCO-pretrained classes (car, motorcycle, bus, truck, person).
- **Wheels:** ~150 frames labelled in Roboflow with SAM-assisted masks.
- **Geometry:** OpenCV `fitEllipse` on wheel masks; the waterline is where the water mask cuts the ellipse.

### 4.2 Depth estimation
```
h_r  = f_r · L_r
σ_r² = (L_r · σ_f)² + (f_r · σ_L)²

ĥ    = Σ (h_r / σ_r²) / Σ (1 / σ_r²)
σ_ĥ² = 1 / Σ (1 / σ_r²)
```
If no reference object is visible, a fallback regressor predicts depth with a deliberately large σ.

### 4.3 Forecaster
- **Training data:** thousands of synthetic rise-and-drain curves, `h(t) = h∞ + (h₀ − h∞)e^(−kt)`, with random k, rain pulses and noise, plus a few real tracked series.
- **Loss:** pinball (quantile) loss, τ ∈ {0.1, 0.5, 0.9}.
- **Baselines:** (1) physics curve fitted by least squares alone, (2) pure GRU, (3) hybrid physics + GRU (ours).

### 4.4 Vehicle safe depths (indicative, configurable)

| Vehicle class | Safe depth |
|---|---|
| Pedestrian | ≈ 15 cm |
| Two-wheeler | ≈ 20 cm |
| Auto-rickshaw | ≈ 20 cm |
| Hatchback / sedan | ≈ 25–30 cm |
| SUV | ≈ 45 cm |
| Bus / truck | ≈ 50 cm |

---

## 5. Object-oriented analysis and design

### 5.1 Actors

| Actor | Type | Goal |
|---|---|---|
| Commuter | Primary | Safe route + wait-or-detour advice for their vehicle |
| Delivery rider / Fleet operator | Primary | Route riders safely with minimal lost time |
| Citizen reporter | Primary | Submit a photo of waterlogging |
| Municipal operator | Primary | See hotspots, verify reports, override segment state |
| System admin | Primary | Register cameras, map them to segments, tune thresholds |
| Camera network | Secondary (system) | Supplies video streams |
| Weather service | Secondary (system) | Supplies rainfall observations and forecasts |

### 5.2 Use cases
1. Set vehicle profile
2. Plan safe route *(includes: Forecast recession / time-to-passable)*
3. Get wait-or-detour advice *(extends: Plan safe route)*
4. View live waterlogging map
5. Report waterlogging (photo) *(includes: Estimate water depth)*
6. Ingest camera stream *(includes: Estimate water depth)*
7. Estimate water depth
8. Forecast recession / time-to-passable
9. Verify / moderate reports
10. Receive hotspot alerts
11. Manage cameras & thresholds
12. Override segment state *(municipal operator)*

Detailed specifications: **UC-01 Plan safe route**, **UC-05 Report waterlogging**, **UC-09 Verify reports**.

### 5.3 Core classes

| Class | Responsibility |
|---|---|
| `RoadSegment` | Holds tracked depth and current `SegmentState`; notifies observers; `timeToPassable(v)` |
| `SegmentState` (interface) → `Dry`, `Wet`, `Rising`, `Receding`, `Unknown` | Per-state behaviour (re-forecast rate, fail-safe rules) |
| `VehicleProfile` (abstract) → `Pedestrian`, `TwoWheeler`, `Auto`, `Car`, `SUV`, `Bus` | `safeDepthCm`, `isPassable(h, σ)` |
| `DepthEstimator` (interface) → `ReferenceObjectEstimator`, `LearnedDepthEstimator` | `estimate(frame) → Observation` |
| `Observation` (abstract) → `CameraObservation`, `CitizenReport` | timestamp, depthCm, sigmaCm, reliability |
| `SegmentStateTracker` | Kalman filter; `assimilate(obs)` |
| `RecessionForecaster` | `forecast(history, rain) → Quantiles` |
| `RoutePlanner` | `plan(o, d, vehicle, t0) → Route`; `waitOrDetour() → Advice` |
| `SegmentObserver` (interface) → `AlertService`, `RouteCacheInvalidator` | `onStateChange(seg)` |
| `CameraSource` (interface) → `RtspAdapter`, `VideoFileAdapter`, `PhotoAdapter` | One frame interface for every source |
| `ReportValidator` chain → `FormatCheck` → `DuplicateCheck` → `QualityCheck` → `ConfidenceCheck` → `HumanModeration` | Citizen report validation |
| `VehicleProfileFactory`, `EstimatorFactory` | Create objects from config |
| `JalMargService` | Facade used by the API / UI |
| `SegmentRepository`, `ObservationRepository` | Storage abstraction |

### 5.4 RoadSegment state machine

```
        rain > r0             h > 5 cm
 Dry ───────────────► Wet ───────────────► Rising
  ▲   dry for 30 min   ▲                    │  ▲
  └────────────────────┘                    │  │ rain resumes
                       ▲      h < 5 cm      ▼  │
                       └──────────────── Receding
                                       (dh/dt < 0 sustained)

 any state ── no observation > 20 min ──► Unknown ── new observation ──► Rising / Receding / Wet
```

> **Design decision:** *Impassable* is **not** a state. A segment is in one state at a time, but passability depends on the vehicle, so it's a query: `segment.timeToPassable(vehicle)`. In `Unknown`, segments are treated as impassable for vehicles with safe depth < 30 cm (fail-safe).

### 5.5 Design patterns

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `VehicleProfile`, `DepthEstimator` | Add vehicle types / estimators without changing the router or tracker (Open/Closed) |
| **Observer** | `RoadSegment` → `AlertService`, `RouteCacheInvalidator`, dashboard | Segment doesn't need to know who's listening |
| **State** | `SegmentState` hierarchy | Behaviour depends on state; no long if/else chains |
| **Adapter** | `CameraSource` | Perception sees one interface for RTSP, video file and photo |
| **Chain of Responsibility** | `ReportValidator` chain | Each check can pass, reject or escalate; easy to reorder |
| **Factory** | `VehicleProfileFactory`, `EstimatorFactory` | Build objects from config or user choice |
| **Facade** | `JalMargService` | UI and API talk to one simple interface |
| **Repository** | `SegmentRepository`, `ObservationRepository` | Domain logic doesn't depend on storage; SQLite now, PostGIS later |

### 5.6 Layered architecture

| Layer | Contents |
|---|---|
| Presentation | Streamlit + Folium map, municipal dashboard view |
| Application / API | `JalMargService` facade (FastAPI: `/route`, `/segments`, `/reports`, `/cameras`) |
| Domain | `RoadSegment`, states, `VehicleProfile`, `Observation`, `RoutePlanner` (pure Python, unit-tested) |
| ML services | Perception + depth (PyTorch / Ultralytics), forecaster |
| Infrastructure | SQLite repositories, camera adapters, Open-Meteo client |

---

## 6. Repository structure

```
jalmarg/
├── README.md
├── requirements.txt
├── data/
│   ├── roadway/          # Roadway Flooding Image Dataset (Kaggle)
│   ├── floodarea/        # Flood Area Segmentation (Kaggle)
│   ├── wheels/           # Roboflow export (YOLO-seg format)
│   ├── clips/            # Indian flood test clips (not redistributed)
│   └── synthetic/        # Generated depth time series
├── jalmarg/
│   ├── domain/           # segment.py, states.py, vehicles.py, observation.py, observers.py
│   ├── routing/          # graph.py, td_dijkstra.py, planner.py
│   ├── perception/       # segmenter.py, wheel_geometry.py, depth_fusion.py
│   ├── temporal/         # kalman.py, forecaster.py, synth.py
│   ├── adapters/         # camera_source.py, weather.py
│   ├── validation/       # report_chain.py
│   ├── repositories/     # sqlite_repos.py
│   └── service.py        # JalMargService facade
├── notebooks/            # training + evaluation (Colab)
├── app/                  # streamlit_app.py
├── uml/                  # PlantUML sources (.puml) + exported PNGs
├── tests/                # pytest, domain layer first
└── reports/              # OOAD report, DL report, demo video
```

---

## 7. Setup

```bash
# Python packages
pip install ultralytics opencv-python torch scipy pandas filterpy osmnx networkx \
            streamlit folium streamlit-folium pytest kaggle yt-dlp

# Datasets (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d saurabhshahane/roadway-flooding-image-dataset -p data/roadway --unzip
kaggle datasets download -d faizalkarim/flood-area-segmentation -p data/floodarea --unzip

# Test clips (research use only, do not redistribute)
yt-dlp -f "best[height<=720]" -o "data/clips/%(title)s.%(ext)s" <url>

# Run
pytest tests/
streamlit run app/streamlit_app.py
```

---

## 8. 10-day plan

| Day | Main work | In the background |
|---|---|---|
| 1 | Fix UML issues, SRS, repo setup | Dataset downloads |
| 2 | Domain layer: states, vehicles, observers, facade + unit tests | Roboflow wheel labelling |
| 3 | Fine-tune YOLO11-seg on Colab; synthetic series generator | Training |
| 4 | Wheel geometry depth + fusion; evaluate on labelled frames | Training |
| 5 | Forecaster: physics baseline, GRU, hybrid (pinball loss) | GRU training |
| 6 | Kalman tracker, time-to-passable, connect ML to domain | |
| 7 | OSMnx graph, time-dependent Dijkstra, Streamlit demo | |
| 8 | Remaining UML: sequence ×2, activity, component, deployment; UC specs | |
| 9 | OOAD + DL reports, results tables, demo video | |
| 10 | Buffer, viva prep | |

**Rule:** always have something training while you write diagrams or reports.

---

## 9. Success metrics

| Component | Metric | Target |
|---|---|---|
| Water segmentation | mIoU on held-out set | ≥ 0.80 |
| Wheel detection | mAP@0.5 | ≥ 0.75 |
| Depth estimation | MAE (cm) | ≤ 5 synthetic, ≤ 8 real |
| Uncertainty | % true depths inside 80% interval | 70–90% |
| Forecaster | MAE at 10 / 30 / 60 min; interval coverage | Beats physics-only and GRU-only |
| Time-to-passable | **Unsafe rate** (said safe, wasn't) | ≈ 0 |
| Routing | Unsafe edges used; time saved vs naive | 0; shown in demo |
| System | Latency per frame on laptop | ≤ 1 s |

**The unsafe rate is the most important number.** Being too cautious sometimes is acceptable; saying "go" into deep water is not.

---

## 10. Deliverables

| OOAD | Deep Learning |
|---|---|
| SRS (functional + non-functional) | Dataset card: sources, licences, splits |
| Use case diagram + 3 detailed specs | Perception: training curves, mIoU, mAP, error analysis |
| Class diagram, 2–3 sequence diagrams, state machine, activity diagram | Depth module: MAE, calibration plot, failure cases |
| Component + deployment diagrams | Forecaster vs baselines at 10/30/60 min |
| Design pattern justification table | Ablation: with/without physics prior, with/without fusion |
| Working domain code with unit tests | Demo video + technical report |

---

## 11. Safety and ethics

- **Advisory only:** "Likely passable for scooters in ~18 min (estimate)", never "safe". Visible disclaimer.
- **Conservative:** decisions use the 90th-percentile forecast plus a safety margin.
- **Fail-safe:** stale or missing data counts as impassable for low vehicles.
- **Human in the loop:** municipal overrides beat the model.
- **Stated limits:** fast currents, open manholes and road collapse can't be seen in water depth.
- **Privacy:** only depths and masks are stored; faces and plates are blurred.

---

## 12. Viva quick answers

| Question | Answer |
|---|---|
| Isn't depth-from-images already done? | Yes, and we build on it. Our contribution is what comes after: per-location forecasting, per-vehicle time-to-passable, wait-or-detour routing. |
| Why not one big end-to-end network? | Too little real labelled data. The physics prior and geometric depth make it data-efficient, explainable and testable. |
| How do you know the wheel size? | Class-based ranges (car vs two-wheeler), carried into σ. Several references reduce error. |
| What if the model is wrong? | Pessimistic forecast + margin, missing data = unsafe, advisory only. Unsafe rate is the key metric. |
| Why Kalman and not averaging? | Weights each source by uncertainty, tracks rate of change, handles gaps. |
| Why is Impassable not a state? | A segment has one state at a time, but passability depends on the vehicle, so it's a query on (segment, vehicle). |
| Why Blender was dropped | The forecaster only needs depth numbers over time, which can be generated numerically from the physics model. |

---

## References

1. Floodwater depth prediction on roadways from side-view real and synthetic images of vehicles. *Data Science for Transportation*, Springer, 2022.
2. BEW-YOLOv8: multi-scene and multi-scale flood depth estimation. *Journal of Hydrology*, 2024.
3. Urban flood-depth estimation using reference-object reasoning and conditional fusion. *ISPRS IJGI* (MDPI), 2026.
4. Vision-based environmental sensing for flood risk forecasting. *Sensors*, 2026.
5. Scalable flood level trend monitoring with surveillance cameras. *HESS*, 2019.
6. Roadway Flooding Image Dataset. Kaggle / Mendeley Data.
7. Flood Area Segmentation. Kaggle.
8. US National Weather Service, "Turn Around Don't Drown".
9. Open-Meteo API · OSMnx · Ultralytics YOLO · Roboflow.
10. The Patents Act, 1970 (India), Section 3(k).

*Verify numbers quoted from papers against the originals before citing them in reports.*
