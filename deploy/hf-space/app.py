"""Hugging Face Space (Gradio SDK, free tier) serving the Pack Station API with the trained models.

The REST API is the same FastAPI app as deploy/vercel-api (`/readyz`, `POST /v1/detect`). A small
Gradio page is mounted at `/` for trying a photo in the browser.

Which model release is served is decided by a release file, `release.json` on the repository's
`model-assets` branch (override with PVS_RELEASE_URL). It names the release, the config to load and
every file to fetch, each pinned by SHA-256:

    {"release": "v1-hosted-onnx-demo", "config": "configs/station.hosted.yaml",
     "files": [{"url": "...tgz", "sha256": "...", "extract": true},
               {"url": "...onnx", "sha256": "...", "path": "models/trained/x.onnx"}]}

On start-up the Space loads the release named there. A background thread re-reads the file every
PVS_POLL_S seconds (default 600). When the release name changes, the new files are downloaded and
verified, a new pipeline is built, and requests switch over only once it is ready. Any failure keeps
the release that is already serving. So a new model is deployed, or rolled back, by pushing
`release.json`; the Space itself does not need to be edited or restarted.

If the release file can't be read at start-up, the built-in v1 release below is served.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

try:  # ZeroGPU hardware expects the `spaces` package; inference here runs on CPU (ONNX Runtime)
    import spaces  # noqa: F401
except Exception:  # noqa: BLE001 - absent outside Hugging Face
    spaces = None

RELEASE_URL = os.environ.get(
    "PVS_RELEASE_URL",
    "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/release.json")
POLL_S = int(os.environ.get("PVS_POLL_S", "600"))
ROOT = Path(os.environ.get("PVS_ROOT", "/tmp/pvs"))
os.environ.setdefault("PVS_CORS_ORIGINS", "https://pack-station-psi.vercel.app")

BUILTIN_RELEASE = {
    "release": "v1-hosted-onnx-demo",
    "config": "configs/station.hosted.yaml",
    "files": [{
        "url": "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/pvs-hosted-bundle.tgz",
        "sha256": "8ba6fdaa22773af1dd0e144b239ec825689f8cad10e844f8912cfcdd11fc6787",
        "extract": True,
    }],
}


def log(*a) -> None:
    print("[release]", *a, flush=True)


def fetch_release() -> dict:
    req = urllib.request.Request(RELEASE_URL, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https URL
        rel = json.loads(r.read())
    if not isinstance(rel.get("release"), str) or not rel["release"] or "/" in rel["release"]:
        raise ValueError("release.json: bad 'release' name")
    if not isinstance(rel.get("config"), str) or not rel.get("files"):
        raise ValueError("release.json: needs 'config' and 'files'")
    for f in rel["files"]:
        if not (f.get("url", "").startswith("https://") and len(f.get("sha256", "")) == 64):
            raise ValueError("release.json: every file needs an https url and a sha256")
        if not f.get("extract") and not f.get("path"):
            raise ValueError("release.json: a file needs 'extract': true or a 'path'")
    return rel


def _download(url: str, sha256: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as out:  # noqa: S310 - https only
        while chunk := r.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {url}: {h.hexdigest()}")
    tmp.replace(dest)


def materialise(rel: dict) -> Path:
    """Download and verify every file of a release into its own folder. Returns that folder."""
    d = ROOT / "releases" / rel["release"]
    marker = d / ".complete"
    if marker.exists() and marker.read_text() == json.dumps(rel, sort_keys=True):
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    try:
        _fetch_files(rel, d)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    marker.write_text(json.dumps(rel, sort_keys=True))
    return d


def _fetch_files(rel: dict, d: Path) -> None:
    for i, f in enumerate(rel["files"]):
        if f.get("extract"):
            tgz = ROOT / "downloads" / f"{rel['release']}-{i}.tgz"
            _download(f["url"], f["sha256"], tgz)
            with tarfile.open(tgz, mode="r:gz") as t:
                t.extractall(d, filter="data")
            tgz.unlink(missing_ok=True)
        else:
            target = (d / f["path"]).resolve()
            if not target.is_relative_to(d.resolve()):
                raise ValueError(f"path escapes the release folder: {f['path']}")
            _download(f["url"], f["sha256"], target)


class Current:
    """The release being served. Swapped as a whole once a new release is fully built."""

    def __init__(self) -> None:
        self.release = None
        self.pipeline = None
        self.api = None
        self.lock = threading.Lock()

    def load(self, rel: dict) -> None:
        from pvs.api import create_app
        from pvs.config import load_station
        from pvs.pipeline import Pipeline
        from pvs.schema import Profile

        d = materialise(rel)
        os.chdir(d)
        cfg = load_station(d / rel["config"])
        pipeline = Pipeline(cfg, Profile.STATION, root=d)
        api = create_app(pipeline, cfg)
        with self.lock:
            old = self.release
            self.release, self.pipeline, self.api = rel, pipeline, api
        log(f"serving {rel['release']} (bundle {cfg.bundle_id})")
        if old and old["release"] != rel["release"]:
            for p in (ROOT / "releases").iterdir():  # keep the new and the previous release only
                if p.name not in (rel["release"], old["release"]):
                    shutil.rmtree(p, ignore_errors=True)


CURRENT = Current()


class ApiProxy:
    """ASGI app that forwards every API request to the release currently being served."""

    async def __call__(self, scope, receive, send):
        await CURRENT.api(scope, receive, send)


def start() -> None:
    try:
        rel = fetch_release()
    except Exception as e:  # noqa: BLE001 - fall back to the built-in release
        log("release.json unreadable, using built-in release:", e)
        rel = BUILTIN_RELEASE
    try:
        CURRENT.load(rel)
    except Exception:  # noqa: BLE001
        if rel is BUILTIN_RELEASE:
            raise
        log("release failed to load, using built-in release:\n" + traceback.format_exc())
        CURRENT.load(BUILTIN_RELEASE)


def poll_forever() -> None:
    failed = None
    while True:
        time.sleep(POLL_S)
        try:
            rel = fetch_release()
            if rel == CURRENT.release or rel == failed:
                continue
            log(f"new release {rel['release']}: loading")
            CURRENT.load(rel)
            failed = None
        except Exception:  # noqa: BLE001 - keep serving the current release
            failed = rel if "rel" in locals() else None
            log("update failed, still serving", CURRENT.release["release"], "\n" + traceback.format_exc())


def build_demo():
    import cv2
    import gradio as gr

    from pvs.capture import frame_from_file

    def analyse(path):
        if not path:
            return None, "Choose a photo."
        pipeline = CURRENT.pipeline
        img = cv2.imread(path)
        s = min(1.0, 1600 / max(img.shape[:2]))
        if s < 1:
            path = str(ROOT / "upload.jpg")
            cv2.imwrite(path, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        res = pipeline.process(frame_from_file(path)).model_dump(mode="json")
        out = cv2.imread(path)
        for i in res["instances"]:
            x0, y0, x1, y1 = map(int, i["bbox_xyxy_px"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (115, 122, 15), 3)
            ax = i["pose"]["axis"]
            ident = i["identity"]
            name = ident.get("sku") or ident.get("family") or ident["level"]
            label = f"{name}" + (f' axis {ax["value_deg"]:.1f}' if ax["available"] else "")
            cv2.putText(out, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (115, 122, 15), 2)
        summary = {k: res[k] for k in ("status", "profile", "model_bundle", "timing_ms") if k in res}
        summary["instances"] = [{"bbox": i["bbox_xyxy_px"], "identity": i["identity"]["level"],
                                 "family": i["identity"].get("family"),
                                 "axis_deg": i["pose"]["axis"].get("value_deg")} for i in res["instances"]]
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), json.dumps(summary, indent=1)

    with gr.Blocks(title="Pack Station API") as demo:
        gr.Markdown("**Pack Station API.** This page runs the trained models served by this Space (ONNX on CPU). "
                    "The REST API for the website is at `/v1/detect`; `/readyz` names the model bundle in use.")
        with gr.Row():
            inp = gr.Image(type="filepath", label="Photo")
            out = gr.Image(label="Result")
        txt = gr.Code(language="json", label="Summary")
        gr.Button("Analyse").click(analyse, inp, [out, txt], api_name=False)
    return demo


def attach_api(demo) -> None:
    """Serve the REST API from Gradio's own server, however the Space runtime launches `demo`.

    Gradio builds its FastAPI app in `App.create_app`; the wrapper below prepends one route per API path,
    each forwarding to the pvs FastAPI app of the current release (which keeps its CORS middleware and handlers)."""
    from gradio.routes import App
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Route

    skip = {"/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    paths = sorted({r.path for r in CURRENT.api.router.routes if getattr(r, "path", "/") not in skip})
    orig = App.create_app
    proxy = ApiProxy()

    def create_app(blocks, *args, **kwargs):
        app = orig(blocks, *args, **kwargs)
        if blocks is demo:
            app.router.routes[0:0] = [Route(p, endpoint=proxy, methods=None) for p in paths]
            # outermost, so cross-origin preflights from the website get the API's CORS policy
            origins = [o.strip() for o in os.environ["PVS_CORS_ORIGINS"].split(",") if o.strip()]
            app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                               allow_headers=["Authorization", "Content-Type"])
            print("API routes attached:", paths, flush=True)
        return app

    App.create_app = staticmethod(create_app)


start()
demo = build_demo()
attach_api(demo)
threading.Thread(target=poll_forever, name="release-poll", daemon=True).start()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", ssr_mode=False)
