"""FastAPI facade over JalMargService: /route, /segments, /reports, /cameras.

    uvicorn jalmarg.api:app --reload
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .demo.scenario import START, replay
from .domain.observation import CitizenReport

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
