"""retail.download with fork=True against a fake Roboflow SDK (no network)."""
import json
import sys
import types
from pathlib import Path

import yaml

from pvs.training import retail

LIMIT = 'RoboflowError: {"error":{"code":"foreign_export_weekly_limit_exceeded","message":"limited to 20"}}'


class FakeVersion:
    def __init__(self, n, licence="CC BY 4.0"):
        self.version, self.licence = f"x/{n}", licence

    def download(self, fmt, location, overwrite):
        d = Path(location)
        (d / "train").mkdir(parents=True, exist_ok=True)
        (d / "train" / "_annotations.coco.json").write_text("{}")
        (d / "README.roboflow.txt").write_text(f"License: {self.licence}\n")


class World:
    """Source projects (other workspaces) and our own workspace 'me'."""

    def __init__(self):
        self.calls = []
        self.own = {"me/old-proj-ab12c": [FakeVersion(1)]}  # an earlier fork, reused
        self.src = {
            "a/direct": [FakeVersion(3)],
            "b/limited": "limit",
            "c/old-proj": [],  # no versions; already forked earlier
            "d/broke": "limit",
            "e/after": "limit",
            "f/badlic": "limit",
        }
        self.licences = {"b/limited": "CC BY 4.0", "f/badlic": "Proprietary"}
        self.credits = {"b/limited"}  # forks that succeed; any other new fork hits 402


def install_fake(monkeypatch, w: World):
    class Proj:
        def __init__(self, pid):
            self.pid = pid

        def versions(self):
            v = w.own[self.pid] if self.pid.startswith("me/") else w.src[self.pid]
            if v == "limit":
                raise RuntimeError(LIMIT)
            return v

        def generate_version(self, settings):
            assert set(settings) == {"preprocessing", "augmentation"}
            w.calls.append(("generate", self.pid))
            w.own[self.pid] = [FakeVersion(1)]
            return 1

        def version(self, n):
            return w.own[self.pid][-1]

    class WS:
        def __init__(self, url):
            self.url = url
            self.project_list = [{"id": k} for k in w.own] if url == "me" else []

        def project(self, name):
            return Proj(f"{self.url}/{name}")

    class RF:
        def __init__(self, api_key):
            pass

        def workspace(self, url=None):
            return WS(url or "me")

    def fork_project(api_key, dest, url=None, source_project_slug=None):
        src = "/".join(url.rstrip("/").split("/")[-2:])
        w.calls.append(("fork", src))
        if src not in w.credits:
            raise RuntimeError('402 {"error": "not enough credits"}')
        return {"taskId": "t-" + src, "url": "https://poll/" + src}

    def poll_until_terminal(api_key, ws, task, polling_url=None, interval=4, timeout=1800):
        src = task[2:]
        slug = src.split("/")[1] + "-zz9yx"
        w.own[f"me/{slug}"] = []
        return {"status": "completed", "result": {"id": f"me/{slug}"}}

    def get_project(api_key, ws, name):
        lic = w.licences.get(f"{ws}/{name}")
        return {"project": {"license": lic}} if lic else {"project": {}}

    rf_mod = types.ModuleType("roboflow")
    rf_mod.Roboflow = RF
    adapters = types.ModuleType("roboflow.adapters")
    rfapi = types.ModuleType("roboflow.adapters.rfapi")
    rfapi.fork_project, rfapi.get_project = fork_project, get_project
    adapters.rfapi = rfapi
    core = types.ModuleType("roboflow.core")
    at = types.ModuleType("roboflow.core.async_tasks")
    at.poll_until_terminal = poll_until_terminal
    for k, v in {"roboflow": rf_mod, "roboflow.adapters": adapters, "roboflow.adapters.rfapi": rfapi,
                 "roboflow.core": core, "roboflow.core.async_tasks": at}.items():
        monkeypatch.setitem(sys.modules, k, v)


def test_fork_path(tmp_path, monkeypatch):
    w = World()
    install_fake(monkeypatch, w)
    lst = tmp_path / "list.yaml"
    lst.write_text(yaml.safe_dump({"allowed_licences": ["CC BY 4.0"], "projects": [
        {"ws": s.split("/")[0], "project": s.split("/")[1]} for s in w.src]}))
    out = tmp_path / "rf"
    res = retail.download(lst, out, "key", fork=True, fork_workers=1)
    rep = {f"{r['ws']}/{r['project']}": r for r in json.loads((out / "download_report.json").read_text())}

    assert rep["a/direct"]["ok"] and "via" not in rep["a/direct"]
    b = rep["b/limited"]
    assert b["ok"] and b["via"] == "fork" and b["fork"] == "me/limited-zz9yx" and b["generated_version"]
    assert "foreign_export" in b["direct_skip"] and "skip" not in b
    assert (out / "b__limited" / "train" / "_annotations.coco.json").exists()
    c = rep["c/old-proj"]  # existing fork reused, never forked again
    assert c["ok"] and c["fork"] == "me/old-proj-ab12c" and ("fork", "c/old-proj") not in w.calls
    assert not rep["d/broke"]["ok"] and "402" in rep["d/broke"]["skip"]
    assert not rep["e/after"]["ok"] and "forking stopped" in rep["e/after"]["skip"]
    assert ("fork", "e/after") not in w.calls  # no more fork calls after running out of credits
    f = rep["f/badlic"]  # licence known up front: rejected without spending a fork
    assert not f["ok"] and f["skip"] == "licence not allowed: Proprietary" and ("fork", "f/badlic") not in w.calls
    assert res["ok"] == 3 and res["via_fork"] == 2


def test_no_fork_flag_keeps_old_behaviour(tmp_path, monkeypatch):
    w = World()
    install_fake(monkeypatch, w)
    lst = tmp_path / "list.yaml"
    lst.write_text(yaml.safe_dump({"allowed_licences": ["CC BY 4.0"],
                                   "projects": [{"ws": "a", "project": "direct"}, {"ws": "b", "project": "limited"}]}))
    res = retail.download(lst, tmp_path / "rf", "key")
    assert res["ok"] == 1 and not any(c[0] == "fork" for c in w.calls)
