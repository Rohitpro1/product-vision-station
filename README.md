# model-assets

Binary runtime assets for Rohitpro1/product-vision-station, kept off `main`.

- `pvs-hosted-bundle.tgz`: the runtime bundle fetched by `deploy/vercel-api/app.py`. Built by `scripts/build_hosted_bundle.sh` at `7453926`.
  - Contains the RT-DETR-R18 package detector ONNX (SHA-256 `68782ed8…cde06`), the hosted config, thresholds and the demo catalogue.
  - Bundle SHA-256: `8ba6fdaa22773af1dd0e144b239ec825689f8cad10e844f8912cfcdd11fc6787`
- The detector was trained on public Roboflow retail photos (CC BY 4.0), starting from PekingU/rtdetr_r18vd (Apache-2.0).
- Results are in `evidence/training/real5star_kaggle_t4x2/` on `main`. 5 Star identity is **not validated**.
