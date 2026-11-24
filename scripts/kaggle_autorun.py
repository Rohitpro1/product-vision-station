"""Kaggle cell driver: run scripts/gpu_train_yolo.sh, and if it fails, wait for a fix on GitHub and resume.

Every stage that already finished in this session is skipped on the next attempt (markers in $WORK), so a fix
pushed to main after a failure costs only the failed stage, not the downloads again. Stops when the fix does not
come in time, when too little of the session is left to be useful, or after MAX_ATTEMPTS.

    env: REPO, WORK, OUT, ROBOFLOW_API_KEY, HF_TOKEN [, TIME_LIMIT_H=12, MAX_ATTEMPTS=4, WAIT_FIX_MIN=120]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

GIT = os.environ.get("AUTORUN_GIT", "https://github.com/Rohitpro1/product-vision-station")
SCRIPT = os.environ.get("AUTORUN_SCRIPT", "scripts/gpu_train_yolo.sh")
POLL_S = float(os.environ.get("AUTORUN_POLL_S", "90"))
REPO = os.environ["REPO"]
LIMIT_H = float(os.environ.get("TIME_LIMIT_H", "12"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "4"))
WAIT_FIX_S = 60 * float(os.environ.get("WAIT_FIX_MIN", "120"))
MIN_LEFT_S = float(os.environ.get("AUTORUN_MIN_LEFT_S", 3 * 3600))  # a retry needs ~2 h of training plus ~1 h of export / tests to be worth it
START = int(os.environ.get("SESSION_START") or time.time())


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd} -> exit {r.returncode}: {r.stderr.strip()[-400:]}")
    return r.stdout.strip()


def remote_head() -> str | None:
    try:
        return sh(f"git ls-remote {GIT} refs/heads/main").split()[0]
    except Exception as e:  # noqa: BLE001 - network blip: try again next poll
        print(f"[autorun] cannot read GitHub: {e}", flush=True)
        return None


def update_repo() -> str:
    if not os.path.isdir(os.path.join(REPO, ".git")):
        sh(f"git clone --depth 1 {GIT} {REPO}")
    sh(f"git -C {REPO} fetch --depth 1 origin main && git -C {REPO} reset --hard FETCH_HEAD")
    return sh(f"git -C {REPO} rev-parse HEAD")


def run_once() -> int:
    env = dict(os.environ, SESSION_START=str(START))
    p = subprocess.Popen(["bash", os.path.join(REPO, SCRIPT)], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def left_s() -> float:
    return START + LIMIT_H * 3600 - time.time()


for attempt in range(1, MAX_ATTEMPTS + 1):
    sha = update_repo()
    print(f"[autorun] attempt {attempt}/{MAX_ATTEMPTS} on commit {sha[:7]}; {left_s() / 3600:.1f} h of the session left",
          flush=True)
    rc = run_once()
    if rc == 0:
        print("[autorun] finished", flush=True)
        sys.exit(0)
    if attempt == MAX_ATTEMPTS or left_s() < MIN_LEFT_S:
        print(f"[autorun] failed (exit {rc}); no retry ({left_s() / 3600:.1f} h left)", flush=True)
        sys.exit(rc)
    deadline = time.time() + min(WAIT_FIX_S, left_s() - MIN_LEFT_S)
    print(f"[autorun] failed (exit {rc}). Waiting up to {(deadline - time.time()) / 60:.0f} min for a fix on GitHub "
          f"(new commit after {sha[:7]}); finished stages will be skipped.", flush=True)
    while time.time() < deadline:
        head = remote_head()
        if head and head != sha:
            print(f"[autorun] fix found: {head[:7]}", flush=True)
            break
        time.sleep(POLL_S)
    else:
        print("[autorun] no fix arrived in time; stopping", flush=True)
        sys.exit(rc)
