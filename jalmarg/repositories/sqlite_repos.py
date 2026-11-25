"""Repositories: domain code depends on these interfaces, not on storage.
SQLite / in-memory now; PostGIS later without touching the domain."""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from ..domain.observation import CameraObservation, CitizenReport, Observation
from ..domain.segment import RoadSegment


class SegmentRepository(ABC):
    @abstractmethod
    def add(self, seg: RoadSegment) -> None: ...
    @abstractmethod
    def get(self, segment_id: str) -> Optional[RoadSegment]: ...
    @abstractmethod
    def all(self) -> List[RoadSegment]: ...
    def save(self, seg: RoadSegment) -> None:
        """Persist the segment's current snapshot (no-op for in-memory)."""


class ObservationRepository(ABC):
    @abstractmethod
    def add(self, obs: Observation) -> None: ...
    @abstractmethod
    def for_segment(self, segment_id: str, since: Optional[datetime] = None) -> List[dict]: ...


class InMemorySegmentRepository(SegmentRepository):
    def __init__(self):
        self._segs: Dict[str, RoadSegment] = {}

    def add(self, seg):
        self._segs[seg.segment_id] = seg

    def get(self, segment_id):
        return self._segs.get(segment_id)

    def all(self):
        return list(self._segs.values())


class InMemoryObservationRepository(ObservationRepository):
    def __init__(self):
        self.rows: List[dict] = []

    def add(self, obs):
        self.rows.append(_obs_row(obs))

    def for_segment(self, segment_id, since=None):
        return [r for r in self.rows if r["segment_id"] == segment_id and (since is None or r["timestamp"] >= since.isoformat())]


def _obs_row(obs: Observation) -> dict:
    # Only numbers are stored; never the image itself (privacy).
    return {
        "obs_id": obs.obs_id, "segment_id": obs.segment_id, "timestamp": obs.timestamp.isoformat(),
        "depth_cm": obs.depth_cm, "sigma_cm": obs.sigma_cm, "reliability": obs.reliability,
        "source": obs.source,
        "kind": "camera" if isinstance(obs, CameraObservation) else "citizen" if isinstance(obs, CitizenReport) else "other",
    }


class SqliteSegmentRepository(InMemorySegmentRepository):
    """Live segment objects in an identity map; snapshots written to SQLite for the dashboard."""

    def __init__(self, path: str = ":memory:"):
        super().__init__()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS segment_snapshot (
                segment_id TEXT, name TEXT, state TEXT, depth_cm REAL, sigma_cm REAL,
                rate_cm_min REAL, at TEXT, override TEXT)"""
        )

    def add(self, seg):
        super().add(seg)
        self.save(seg)

    def save(self, seg):
        s = seg.snapshot()
        self.conn.execute(
            "INSERT INTO segment_snapshot VALUES (?,?,?,?,?,?,?,?)",
            (s["segment_id"], s["name"], s["state"], s["depth_cm"], s["sigma_cm"], s["rate_cm_min"],
             seg.now.isoformat() if seg.now != datetime.min else None, s["override"]),
        )
        self.conn.commit()

    def history(self, segment_id: str) -> List[tuple]:
        return self.conn.execute(
            "SELECT at, state, depth_cm FROM segment_snapshot WHERE segment_id=? ORDER BY rowid", (segment_id,)
        ).fetchall()


class SqliteObservationRepository(ObservationRepository):
    COLS = ("obs_id", "segment_id", "timestamp", "depth_cm", "sigma_cm", "reliability", "source", "kind")

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS observation (obs_id TEXT PRIMARY KEY, segment_id TEXT, timestamp TEXT, "
            "depth_cm REAL, sigma_cm REAL, reliability REAL, source TEXT, kind TEXT)"
        )

    def add(self, obs):
        row = _obs_row(obs)
        self.conn.execute(f"INSERT OR REPLACE INTO observation VALUES ({','.join('?' * len(self.COLS))})",
                          tuple(row[c] for c in self.COLS))
        self.conn.commit()

    def for_segment(self, segment_id, since=None):
        q = "SELECT * FROM observation WHERE segment_id=?"
        args: Iterable = (segment_id,)
        if since is not None:
            q += " AND timestamp >= ?"
            args = (segment_id, since.isoformat())
        rows = self.conn.execute(q + " ORDER BY timestamp", tuple(args)).fetchall()
        return [dict(zip(self.COLS, r)) for r in rows]
