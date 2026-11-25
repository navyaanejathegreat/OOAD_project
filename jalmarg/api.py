"""FastAPI facade over JalMargService: /route, /segments, /reports, /cameras.

    uvicorn jalmarg.api:app --reload
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from functools import lru_cache

from .demo.scenario import DURATION_MIN, RAIN_MM_H, RAIN_STOP_MIN, START, replay, true_depths
from .domain.observation import CitizenReport
from .domain.vehicles import VehicleProfileFactory
from .routing.graph import build_demo_graph
from .routing.planner import DISCLAIMER
from .temporal.forecaster import PhysicsForecaster, default_forecaster

app = FastAPI(title="JalMarg API", version="0.1.0",
              description="Advisory estimates only. Not a safety guarantee.")

# Demo: start from the scripted scenario at minute 40 of the storm.
service = replay(40)


class ReportIn(BaseModel):
    segment_id: str
    depth_cm: float = Field(ge=0, le=300)
    sigma_cm: float = Field(default=6.0, gt=0)
    reporter_id: str
    photo_sha256: Optional[str] = None
    note: str = ""
    timestamp: Optional[datetime] = None


class CameraIn(BaseModel):
    camera_id: str
    segment_id: str


class OverrideIn(BaseModel):
    mode: Optional[str] = Field(default=None, description="'open', 'closed' or null to clear")
    by: str = "operator"
    minutes: Optional[float] = None
    reason: str = ""


@app.get("/route")
def route(vehicle: str, origin: str = "home", dest: str = "office", minute: Optional[int] = None):
    t0 = START + timedelta(minutes=minute) if minute is not None else service.now
    try:
        return service.plan_route(vehicle, origin, dest, t0).to_dict(service.graph)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/segments")
def segments(vehicle: Optional[str] = None):
    return service.segment_snapshots(vehicle)


@app.post("/segments/{segment_id}/override")
def override(segment_id: str, body: OverrideIn):
    try:
        service.override_segment(segment_id, body.mode, body.by, body.minutes, body.reason)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    return service.segments.get(segment_id).snapshot()


@app.post("/reports")
def submit_report(body: ReportIn):
    ts = body.timestamp or service.now
    digest = body.photo_sha256 or hashlib.sha256(f"{body.reporter_id}{ts}".encode()).hexdigest()
    r = CitizenReport(body.segment_id, ts, body.depth_cm, body.sigma_cm, reporter_id=body.reporter_id,
                      image_hash=digest, note=body.note)
    res = service.submit_report(r)
    return {"report_id": r.obs_id, "verdict": res.verdict.value, "reasons": res.reasons}


@app.get("/reports/pending")
def pending():
    return [{"report_id": r.obs_id, "segment_id": r.segment_id, "depth_cm": r.depth_cm, "note": r.note}
            for r in service.pending_reports()]


@app.post("/reports/{report_id}/moderate")
def moderate(report_id: str, approve: bool):
    r = service.moderate_report(report_id, approve)
    if r is None:
        raise HTTPException(404, "No such pending report")
    return {"report_id": report_id, "approved": approve}


@app.get("/cameras")
def cameras():
    return service.cameras


@app.post("/cameras")
def register_camera(body: CameraIn):
    try:
        service.register_camera(body.camera_id, body.segment_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return service.cameras


@app.get("/alerts")
def alerts():
    return [a.__dict__ for a in service.alerts.alerts[-50:]]


# ---------------------------------------------------------------- demo replay (used by the React app)
@lru_cache(maxsize=4)
def _forecaster(model: str):
    return PhysicsForecaster() if model == "physics" else default_forecaster()


@lru_cache(maxsize=64)
def _replayed(minute: int, model: str):
    return replay(minute, forecaster=_forecaster(model))


@app.get("/demo/meta")
def demo_meta():
    """Static facts the demo UI needs once: vehicles, timeline and road geometry."""
    g = build_demo_graph()
    roads, seen = [], set()
    for u, v, d in g.G.edges(data=True):
        key = tuple(sorted((str(u), str(v))))
        if key in seen:
            continue
        seen.add(key)
        roads.append({"from": str(u), "to": str(v), "name": d.get("name", ""),
                      "segment_id": d.get("segment_id"), "coords": [g.coords(u), g.coords(v)]})
    places = [{"id": str(n), "name": g.name(n), "coords": g.coords(n)} for n in g.nodes()]
    return {
        "vehicles": [{"key": v.key, "label": v.label, "safe_depth_cm": v.safe_depth_cm,
                      "threshold_cm": v.threshold_cm} for v in VehicleProfileFactory().all()],
        "duration_min": DURATION_MIN, "max_minute": DURATION_MIN - 61, "rain_stop_min": RAIN_STOP_MIN,
        "rain_mm_h": RAIN_MM_H, "start": START.isoformat(timespec="minutes"),
        "roads": roads, "places": places, "disclaimer": DISCLAIMER,
    }


@app.get("/demo/state")
def demo_state(vehicle: str = "two_wheeler", minute: int = 40, model: str = "hybrid",
               origin: str = "home", dest: str = "office"):
    """Everything the demo shows for one vehicle at one moment of the replayed storm."""
    if not 0 <= minute <= DURATION_MIN - 61:
        raise HTTPException(400, f"minute must be between 0 and {DURATION_MIN - 61}")
    if model not in ("hybrid", "physics"):
        raise HTTPException(400, "model must be 'hybrid' or 'physics'")
    try:
        veh = VehicleProfileFactory().create(vehicle)
    except ValueError as e:
        raise HTTPException(400, str(e))
    svc = _replayed(minute, model)
    t0 = START + timedelta(minutes=minute)
    advice = svc.plan_route(vehicle, origin, dest, t0)
    truth = true_depths()
    segs = []
    for s in svc.segments.all():
        ttp = s.time_to_passable(veh, t0)
        q = s.forecast(t0)
        tr = truth[s.segment_id]
        hist = list(range(max(0, minute - 60), minute + 1))
        segs.append({
            **s.snapshot(),
            "time_to_passable": {"status": ttp.status, "minutes": ttp.minutes, "text": ttp.describe(veh.label)},
            "history": [{"minute": m, "true_cm": round(float(tr[m]), 2)} for m in hist],
            "forecast": None if q is None else [
                {"minute": minute + int(mm), "q10": round(float(a), 2), "q50": round(float(b), 2),
                 "q90": round(float(c), 2),
                 "true_cm": round(float(tr[minute + int(mm)]), 2) if minute + int(mm) < len(tr) else None}
                for mm, a, b, c in zip(q.minutes, q.q10, q.q50, q.q90)],
        })
    return {
        "minute": minute, "time": t0.isoformat(timespec="minutes"),
        "raining": minute < RAIN_STOP_MIN, "vehicle": {"key": veh.key, "label": veh.label,
                                                        "threshold_cm": veh.threshold_cm},
        "advice": advice.to_dict(svc.graph),
        "naive_unsafe_names": [svc.segments.get(x).name for x in advice.naive_unsafe_segments],
        "segments": segs,
        "alerts": [{"time": a.time.isoformat(timespec="minutes"), "message": a.message}
                   for a in svc.alerts.alerts[-15:][::-1]],
    }
