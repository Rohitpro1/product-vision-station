"""Live progress with ETA for long runs: readable log lines + a machine-readable status file.

Every stage writes `$PVS_STATUS_DIR/status.json` (atomically), which `pvs.training.hub watch` publishes to a
Hugging Face repo page while the run is going. Log lines look like:

    [train] 1,240/9,600 steps (12.9%) | 2.1 steps/s | stage ETA 1h06m (14:32 IST) | run ETA 1h41m | loss 3.81
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

IST = timedelta(hours=5, minutes=30)

# rough minutes for stages after the current one (used only for the whole-run ETA; real timings replace them)
STAGE_PLAN = ["setup", "download", "prepare", "smoke", "train", "export", "brands", "bundle", "e2e", "upload"]
STAGE_GUESS_MIN = {"setup": 6, "download": 25, "prepare": 15, "smoke": 30, "train": 330, "export": 8,
                   "brands": 25, "bundle": 1, "e2e": 10, "upload": 5}


def fmt_dur(s: float | None) -> str:
    if s is None or s != s or s < 0:
        return "?"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s % 60:02d}s" if m else f"{s}s")


def _clock(ts: float) -> str:
    return (datetime.fromtimestamp(ts, UTC) + IST).strftime("%H:%M IST")


def _status_path() -> Path | None:
    d = os.environ.get("PVS_STATUS_DIR")
    return Path(d) / "status.json" if d else None


def read_status() -> dict:
    p = _status_path()
    if p and p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def write_status(st: dict) -> None:
    p = _status_path()
    if not p:
        return
    # several processes (one mask job per GPU) and threads write this file: a temp name of their own each, and a
    # failed status write is never allowed to stop the work it reports on
    try:
        import shutil

        st = dict(st)
        st["disk_free_gb"] = round(shutil.disk_usage(os.environ.get("PVS_DISK_PATH", "/tmp")).free / 1e9, 1)
    except OSError:
        pass
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".status.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(p)
    except OSError as e:
        print(f"[progress] status not written: {type(e).__name__}: {e}", flush=True)


def remaining_after(stage: str) -> float:
    """Seconds of planned work after `stage` (skips smoke once the full run is going)."""
    st = read_status()
    done = set(st.get("stages_done", {}))
    stage = stage.split(":")[0]  # "smoke:train" -> smoke, "prepare:hash" -> prepare
    later = STAGE_PLAN[STAGE_PLAN.index(stage) + 1:] if stage in STAGE_PLAN else []
    return sum(STAGE_GUESS_MIN[s] * 60 for s in later if s not in done and not (s == "smoke" and st.get("full_run")))


class Progress:
    def __init__(self, stage: str, total: int, unit: str = "items", every_s: float = 20.0, enabled: bool = True):
        self.stage, self.total, self.unit, self.every = stage, max(1, int(total)), unit, every_s
        self.enabled = enabled
        self.done = 0
        self.t0 = self.last = time.time()
        self.extra: dict = {}
        if enabled:
            self._emit(force=True)

    def update(self, n: int = 1, **extra) -> None:
        self.done += n
        self.extra.update({k: v for k, v in extra.items() if v is not None})
        if self.enabled and (time.time() - self.last >= self.every or self.done >= self.total):
            self._emit()

    def set_total(self, total: int) -> None:
        self.total = max(1, int(total))

    def _emit(self, force: bool = False) -> None:
        now = time.time()
        self.last = now
        el = now - self.t0
        rate = self.done / el if el > 0 and self.done else None
        eta = (self.total - self.done) / rate if rate else None
        run_eta = (eta or 0) + remaining_after(self.stage) if eta is not None else None
        pct = 100.0 * self.done / self.total
        extra = " | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in self.extra.items())
        line = (f"[{self.stage}] {self.done:,}/{self.total:,} {self.unit} ({pct:.1f}%)"
                + (f" | {rate:.2f} {self.unit}/s" if rate else "")
                + (f" | stage ETA {fmt_dur(eta)} ({_clock(now + eta)})" if eta is not None else "")
                + (f" | run ETA {fmt_dur(run_eta)} ({_clock(now + run_eta)})" if run_eta is not None else "")
                + (f" | {extra}" if extra else ""))
        print(line, flush=True)
        st = read_status()
        st.update(stage=self.stage, line=line, done=self.done, total=self.total, unit=self.unit,
                  pct=round(pct, 2), rate=rate, stage_eta_s=eta, run_eta_s=run_eta,
                  run_eta_at=datetime.fromtimestamp(now + run_eta, UTC).isoformat(timespec="minutes")
                  if run_eta is not None else None,
                  extra=self.extra, updated_at=datetime.now(UTC).isoformat(timespec="seconds"))
        write_status(st)


def stage_name(default: str) -> str:
    return os.environ.get("PVS_STAGE", default)


def stage_event(stage: str, event: str, note: str = "") -> None:
    """Called by the runner between stages: start/done/failed with wall-clock bookkeeping."""
    st = read_status()
    now = time.time()
    st.setdefault("stages_done", {})
    st.setdefault("started_at", datetime.now(UTC).isoformat(timespec="seconds"))
    if event == "start":
        st.update(stage=stage, stage_started=now, line=f"[{stage}] started {note}".strip(), pct=None,
                  stage_eta_s=None, extra={})
        if not stage.startswith("smoke") and stage in ("train", "export", "brands", "bundle", "e2e"):
            st["full_run"] = True
    elif event == "done":
        st["stages_done"][stage] = round(now - st.get("stage_started", now), 1)
        st["line"] = f"[{stage}] done in {fmt_dur(now - st.get('stage_started', now))} {note}".strip()
    elif event in ("failed", "finished"):
        st["final"] = event
        st["line"] = f"run {event}: {note}".strip()
    st["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_status(st)
    print(f"=== {st['line']}", flush=True)


def main(argv=None) -> int:
    import sys

    args = argv or sys.argv[1:]
    stage_event(args[0], args[1], " ".join(args[2:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
