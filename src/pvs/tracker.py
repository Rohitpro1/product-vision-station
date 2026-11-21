"""ByteTrack-style association with per-track, time-bounded evidence.

Two-stage association (high-score detections first, then low-score
detections against remaining tracks) using IoU + Hungarian matching, as in
ByteTrack (Zhang et al., 2022). Track IDs are scoped ``<camera>:<epoch>:<n>``.

Evidence safety rules (build prompt section 4, Tracking):
* evidence is stored per track with the frame and box it was observed on;
* an implausible jump, an ambiguous merge/split, or an epoch reset clears it;
* a repeated identical frame (same content hash) cannot add observations;
* a *delayed* evidence result (e.g. slow OCR) is accepted only if the track
  still owned the same box at the frame the crop was taken *and* has not
  been marked ambiguous since. Otherwise it is discarded, never re-routed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class EvidenceItem:
    kind: str
    value: object
    frame_id: int
    t: float
    box: list[float]


@dataclass
class Track:
    tid: str
    box: list[float]
    last_t: float
    last_frame: int
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    history: dict[int, list[float]] = field(default_factory=dict)  # frame_id -> box
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)
    ambiguous_since_frame: int | None = None
    seen_hashes: set[str] = field(default_factory=set)
    state: dict = field(default_factory=dict)  # smoothers, last published identity etc.

    def predict(self, t: float) -> list[float]:
        dt = max(0.0, t - self.last_t)
        return (np.array(self.box) + self.velocity * min(dt, 0.5)).tolist()


class Tracker:
    def __init__(self, camera: str, epoch: int = 0, high: float = 0.5, low: float = 0.1, match_iou: float = 0.3,
                 max_gap_s: float = 1.0, max_jump: float = 0.6, history_len: int = 90,
                 evidence_gap_s: float = 0.25) -> None:
        self.camera = camera
        self.epoch = epoch
        self.high, self.low = high, low
        self.match_iou = match_iou
        self.max_gap_s = max_gap_s
        self.max_jump = max_jump
        self.evidence_gap_s = evidence_gap_s
        self.history_len = history_len
        self.tracks: list[Track] = []
        self._next = 1
        self.switch_events = 0

    def reset(self, epoch: int) -> None:
        """Reconnect / resolution change: drop all tracks and evidence."""
        self.epoch = epoch
        self.tracks = []
        self._next = 1

    def _new(self, box, t, frame_id) -> Track:
        tr = Track(f"{self.camera}:{self.epoch}:{self._next}", list(box), t, frame_id)
        tr.history[frame_id] = list(box)
        self._next += 1
        self.tracks.append(tr)
        return tr

    def _assign(self, tracks: list[Track], boxes: list[list[float]], t: float):
        if not tracks or not boxes:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        cost = np.ones((len(tracks), len(boxes)))
        for i, tr in enumerate(tracks):
            p = tr.predict(t)
            for j, b in enumerate(boxes):
                cost[i, j] = 1 - iou(p, b)
        r, c = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(r, c) if 1 - cost[i, j] >= self.match_iou]
        mi = {i for i, _ in pairs}
        mj = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mi], [j for j in range(len(boxes)) if j not in mj]

    def update(self, boxes: list[list[float]], scores: list[float], t: float, frame_id: int,
               content_hash: str | None = None) -> list[Track | None]:
        """Returns the track for each input box (None for unconfirmed low-score boxes)."""
        # expire stale tracks
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_gap_s]
        hi = [j for j, s in enumerate(scores) if s >= self.high]
        lo = [j for j, s in enumerate(scores) if self.low <= s < self.high]
        out: list[Track | None] = [None] * len(boxes)
        pairs, un_t, un_hi = self._assign(self.tracks, [boxes[j] for j in hi], t)
        matched: list[tuple[Track, int]] = [(self.tracks[i], hi[j]) for i, j in pairs]
        rem = [self.tracks[i] for i in un_t]
        pairs2, _, _ = self._assign(rem, [boxes[j] for j in lo], t)
        matched += [(rem[i], lo[j]) for i, j in pairs2]
        # ambiguous merge / split: a box overlapping two live tracks
        live = [tr.predict(t) for tr in self.tracks]
        for b in boxes:
            touching = [k for k, p in enumerate(live) if iou(p, b) >= self.match_iou]
            if len(touching) > 1:
                for k in touching:
                    self._invalidate(self.tracks[k], frame_id)
        for tr, j in matched:
            b = boxes[j]
            diag = float(np.hypot(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1]))
            jump = float(np.hypot((b[0] + b[2] - tr.box[0] - tr.box[2]) / 2, (b[1] + b[3] - tr.box[1] - tr.box[3]) / 2))
            if jump > self.max_jump * diag:
                self._invalidate(tr, frame_id)
            elif t - tr.last_t > self.evidence_gap_s:
                # re-association after a detection gap (removal / long occlusion): the object may not be
                # the same one, so its identity and direction evidence must not carry over
                self._invalidate(tr, frame_id)
            dt = max(t - tr.last_t, 1e-3)
            tr.velocity = 0.5 * tr.velocity + 0.5 * (np.array(b) - np.array(tr.box)) / dt
            tr.box, tr.last_t, tr.last_frame = list(b), t, frame_id
            tr.hits += 1
            tr.history[frame_id] = list(b)
            if len(tr.history) > self.history_len:
                tr.history.pop(min(tr.history))
            out[j] = tr
        for j in un_hi:
            out[hi[j]] = self._new(boxes[hi[j]], t, frame_id)
        return out

    def _invalidate(self, tr: Track, frame_id: int) -> None:
        tr.evidence.clear()
        tr.state.pop("identity", None)
        for k in list(tr.state):
            if k.startswith("smoother"):
                tr.state[k].reset()
        tr.ambiguous_since_frame = frame_id

    # -------------------------------------------------------------- evidence
    def add_evidence(self, tr: Track, kind: str, value, frame_id: int, t: float, content_hash: str | None = None
                     ) -> bool:
        if content_hash is not None:
            key = f"{kind}:{content_hash}"
            if key in tr.seen_hashes:
                return False  # identical frame: not an independent observation
            tr.seen_hashes.add(key)
        tr.evidence[kind] = EvidenceItem(kind, value, frame_id, t, list(tr.history.get(frame_id, tr.box)))
        return True

    def add_delayed_evidence(self, tid: str, kind: str, value, frame_id: int, box_at_capture: list[float],
                             t_capture: float) -> bool:
        """Bind a late result to the track that owned ``box_at_capture`` at ``frame_id`` - or drop it."""
        tr = next((x for x in self.tracks if x.tid == tid), None)
        if tr is None:
            return False
        if tr.ambiguous_since_frame is not None and tr.ambiguous_since_frame >= frame_id:
            return False
        owned = tr.history.get(frame_id)
        if owned is None or iou(owned, box_at_capture) < 0.7:
            return False
        tr.evidence[kind] = EvidenceItem(kind, value, frame_id, t_capture, list(owned))
        return True

    def fresh(self, tr: Track, kind: str, now: float, ttl: float) -> EvidenceItem | None:
        e = tr.evidence.get(kind)
        if e is None or now - e.t > ttl:
            return None
        return e
