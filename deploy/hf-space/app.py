"""Hugging Face Space (Gradio SDK, free tier) serving the Pack Station API with the trained detector.

The REST API is the same FastAPI app as deploy/vercel-api (`/readyz`, `POST /v1/detect`). A small
Gradio page is mounted at `/` for trying a photo in the browser. The model bundle (trained ONNX detector,
config, demo catalogue) is fetched SHA-256-pinned on start-up.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import urllib.request
from pathlib import Path

try:  # ZeroGPU hardware expects the `spaces` package; inference here runs on CPU (ONNX Runtime)
    import spaces  # noqa: F401
except Exception:  # noqa: BLE001 - absent outside Hugging Face
    spaces = None

BUNDLE_URL = os.environ.get(
    "PVS_BUNDLE_URL",
    "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/model-assets/pvs-hosted-bundle.tgz")
BUNDLE_SHA256 = os.environ.get("PVS_BUNDLE_SHA256", "8ba6fdaa22773af1dd0e144b239ec825689f8cad10e844f8912cfcdd11fc6787")
ROOT = Path(os.environ.get("PVS_ROOT", "/tmp/pvs"))
os.environ.setdefault("PVS_CORS_ORIGINS", "https://pack-station-psi.vercel.app")


def _ensure_bundle() -> None:
    marker = ROOT / ".bundle-sha256"
    if marker.exists() and marker.read_text() == BUNDLE_SHA256:
        return
    data = urllib.request.urlopen(BUNDLE_URL, timeout=120).read()  # noqa: S310 - fixed https URL
    got = hashlib.sha256(data).hexdigest()
    if got != BUNDLE_SHA256:
        raise RuntimeError(f"bundle sha256 mismatch: {got}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        t.extractall(ROOT, filter="data")
    marker.write_text(BUNDLE_SHA256)


def build():
    import cv2
    import gradio as gr
    from pvs.api import create_app
    from pvs.capture import frame_from_file
    from pvs.config import load_station
    from pvs.pipeline import Pipeline
    from pvs.schema import Profile

    _ensure_bundle()
    os.chdir(ROOT)
    cfg = load_station(ROOT / "configs/station.hosted.yaml")
    pipeline = Pipeline(cfg, Profile.STATION, root=ROOT)
    api = create_app(pipeline, cfg)

    def analyse(path):
        if not path:
            return None, "Choose a photo."
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
            label = f'{i["identity"]["level"]}' + (f' axis {ax["value_deg"]:.1f}' if ax["available"] else "")
            cv2.putText(out, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (115, 122, 15), 2)
        summary = {k: res[k] for k in ("status", "profile", "timing_ms")}
        summary["instances"] = [{"bbox": i["bbox_xyxy_px"], "identity": i["identity"]["level"],
                                 "axis_deg": i["pose"]["axis"].get("value_deg")} for i in res["instances"]]
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), json.dumps(summary, indent=1)

    with gr.Blocks(title="Pack Station API") as demo:
        gr.Markdown("**Pack Station API.** This page runs the trained package detector (RT-DETR-R18, ONNX). "
                    "The REST API for the website is at `/v1/detect`. Pack identity is not validated, so expect `unknown`.")
        with gr.Row():
            inp = gr.Image(type="filepath", label="Photo")
            out = gr.Image(label="Result")
        txt = gr.Code(language="json", label="Summary")
        gr.Button("Analyse").click(analyse, inp, [out, txt], api_name=False)
    # the Gradio page takes "/" on the Space; the API keeps every other route
    api.router.routes = [r for r in api.router.routes if getattr(r, "path", None) != "/"]
    return gr.mount_gradio_app(api, demo, path="/")


app = build()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
