"""Bounded SQLite-WAL event outbox with idempotent state-transition events.

Event ID = sha256(station | camera | epoch | track | transition_seq): stable
across retries and restarts. The transition row and the per-track state are
written in one transaction. Delivery is at-least-once; consumers deduplicate
by event ID (``DedupConsumer``). Disk-full policy: when ``max_rows`` pending
events exist the oldest *delivered* rows are pruned first; if still full, new
events are refused, a ``high_water`` alarm is raised and a data-loss counter
is incremented (never silently dropped).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, station TEXT, camera TEXT, epoch INTEGER, track TEXT, seq INTEGER,
  kind TEXT, payload TEXT, captured_at TEXT, created_mono REAL,
  delivered INTEGER DEFAULT 0, attempts INTEGER DEFAULT 0, next_attempt REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS track_state(
  camera TEXT, epoch INTEGER, track TEXT, seq INTEGER, state TEXT, PRIMARY KEY(camera, epoch, track));
CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value INTEGER);
CREATE INDEX IF NOT EXISTS ev_pending ON events(delivered, next_attempt);
"""


class OutboxFull(RuntimeError):
    pass


class Outbox:
    def __init__(self, path: Path, station: str, max_rows: int = 100_000, high_water: float = 0.8) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        self.station = station
        self.max_rows = max_rows
        self.high_water = high_water

    @staticmethod
    def event_id(station: str, camera: str, epoch: int, track: str, seq: int) -> str:
        return hashlib.sha256(f"{station}|{camera}|{epoch}|{track}|{seq}".encode()).hexdigest()[:32]

    def _counter(self, name: str, inc: int = 0) -> int:
        if inc:
            self.db.execute("INSERT INTO counters VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+?",
                            (name, inc, inc))
        row = self.db.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()
        return row[0] if row else 0

    def pending(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events WHERE delivered=0").fetchone()[0]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def alarms(self) -> dict:
        p = self.pending()
        return {"pending": p, "high_water": p >= self.high_water * self.max_rows,
                "data_loss_events": self._counter("refused")}

    def record_transition(self, camera: str, epoch: int, track: str, new_state: dict, kind: str,
                          captured_at: str) -> str | None:
        """Atomically compare state, bump seq and enqueue an event. Returns event id or None (no change)."""
        st = json.dumps(new_state, sort_keys=True)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT seq, state FROM track_state WHERE camera=? AND epoch=? AND track=?",
                                  (camera, epoch, track)).fetchone()
            if row and row[1] == st:
                self.db.execute("COMMIT")
                return None
            if self.count() >= self.max_rows:
                self.db.execute("DELETE FROM events WHERE rowid IN (SELECT rowid FROM events WHERE delivered=1 "
                                "ORDER BY created_mono LIMIT ?)", (max(1, self.max_rows // 10),))
                if self.count() >= self.max_rows:
                    self._counter("refused", 1)
                    self.db.execute("COMMIT")
                    raise OutboxFull("outbox full: event refused, data-loss counter incremented")
            seq = (row[0] + 1) if row else 1
            eid = self.event_id(self.station, camera, epoch, track, seq)
            self.db.execute("INSERT OR IGNORE INTO events(event_id,station,camera,epoch,track,seq,kind,payload,"
                            "captured_at,created_mono) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (eid, self.station, camera, epoch, track, seq, kind, st, captured_at, time.monotonic()))
            self.db.execute("INSERT INTO track_state VALUES(?,?,?,?,?) ON CONFLICT(camera,epoch,track) "
                            "DO UPDATE SET seq=excluded.seq, state=excluded.state", (camera, epoch, track, seq, st))
            self.db.execute("COMMIT")
            return eid
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def deliver(self, send: Callable[[dict], None], batch: int = 100, base_backoff: float = 0.5,
                max_backoff: float = 30.0, now: float | None = None) -> dict:
        """Try to deliver due events. ``send`` raises on failure. Ack happens *after* send succeeds."""
        now = time.monotonic() if now is None else now
        rows = self.db.execute("SELECT event_id,camera,epoch,track,seq,kind,payload,captured_at,attempts,next_attempt "
                               "FROM events WHERE delivered=0 ORDER BY created_mono LIMIT ?", (batch,)).fetchall()
        ok = fail = 0
        for eid, cam, ep, tr, seq, kind, payload, cap, att, nxt in rows:
            if nxt > now:
                break  # head-of-line: never overtake an event that is backing off
            ev = {"event_id": eid, "station": self.station, "camera": cam, "epoch": ep, "track": tr, "seq": seq,
                  "kind": kind, "state": json.loads(payload), "captured_at": cap}
            try:
                send(ev)
            except Exception:  # noqa: BLE001 - any delivery failure is retried with backoff
                fail += 1
                delay = min(max_backoff, base_backoff * (2 ** att))
                self.db.execute("UPDATE events SET attempts=attempts+1, next_attempt=? WHERE event_id=?",
                                (now + delay, eid))
                break  # preserve order: stop on first failure
            self.db.execute("UPDATE events SET delivered=1 WHERE event_id=?", (eid,))
            ok += 1
        return {"delivered": ok, "failed": fail, "pending": self.pending()}

    def iter_events(self) -> Iterator[tuple]:
        yield from self.db.execute("SELECT event_id, track, seq, kind, payload FROM events ORDER BY created_mono")

    def close(self) -> None:
        self.db.close()


class DedupConsumer:
    """Reference consumer: idempotent by event_id."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.applied: list[dict] = []
        self.duplicates = 0

    def __call__(self, ev: dict) -> None:
        if ev["event_id"] in self.seen:
            self.duplicates += 1
            return
        self.seen.add(ev["event_id"])
        self.applied.append(ev)


def transitions(result, ob: Outbox) -> list[str]:
    """Emit events when a track's *published* state changes (identity level/sku, face, heading availability)."""
    out = []
    for inst in result.instances:
        if not inst.track_id:
            continue
        st = {"identity_level": inst.identity.level.value, "sku": inst.identity.sku, "family": inst.identity.family,
              "face": inst.pose.face.value, "heading_available": inst.pose.heading.available}
        eid = ob.record_transition(result.camera, result.stream_epoch, inst.track_id, st, "state_change",
                                   result.captured_at)
        if eid:
            out.append(eid)
    return out
