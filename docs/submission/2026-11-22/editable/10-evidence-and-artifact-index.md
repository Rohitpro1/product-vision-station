# Evidence and artifact index

Use this page to reproduce the review. The ZIP also carries machine-readable registers, the evidence status for each requirement, and captured JSON reports.

| Record | Identifier / contents |
| --- | --- |
| Application | Rohitpro1/product-vision-station at b56c2a781939fc1524a5b9d2e1ba297992d58bfb. The handoff baseline ed64b5e is the parent; the next change adds documentation. |
| Assets | model-assets at b1436253c9387fd5408f355f1eb87383657f42e7. Archive length 74,543,598 bytes. Digest and nested ONNX were independently recomputed. |
| HF Space / release | Rohitpro1/pack-station-api at 465a53b02b56ad19a187164d9ebdb27d5c7f1e86. Its app.py matched deploy/hf-space/app.py. Current provider and catalogue were read from both production /v1/meta endpoints. |
| Training publisher | Rohitpro1/pack-station-training at 55b036781ed354634b5f6cc7b562149632b2a8dc. Source telemetry updated 16:37:56 UTC / 22:07:56 IST. This is a fixed observation of a moving run. |
| Verification | CI run 45037028660; v1 train implementation b5ee790. Fixture and pretrained sealed results at implementation 473b608. Follow evidence links for exact denominators and failure slices. |
| Archive contents | Individual PDF records, editable text and diagram sources; dataset, code, model, requirement and source registers; v1 reports, schema, CI and hosted metadata, v2 progress, and SHA-256 file inventory. |

## Verified binary digests

```bash
Archive SHA-256
8ba6fdaa22773af1dd0e144b239ec825689f8cad10e844f8912cfcdd11fc6787
Detector ONNX SHA-256
68782ed8d550fa47334512634239eec067a032063e85ea7ef1fcb697560cde06
```

Source links below identify the principal records. Design and brief links on page 1 require owner access; the repository and evidence registers contain the public implementation references.

Sources: [Main](https://github.com/Rohitpro1/product-vision-station/tree/b56c2a781939fc1524a5b9d2e1ba297992d58bfb), [Assets](https://github.com/Rohitpro1/product-vision-station/tree/b1436253c9387fd5408f355f1eb87383657f42e7), [Space](https://huggingface.co/spaces/Rohitpro1/pack-station-api/tree/465a53b02b56ad19a187164d9ebdb27d5c7f1e86), [Hub](https://huggingface.co/Rohitpro1/pack-station-training/tree/55b036781ed354634b5f6cc7b562149632b2a8dc), [CI](https://github.com/Rohitpro1/product-vision-station/actions/runs/45037028660), [V1 evaluation](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/training/real5star_kaggle_t4x2/eval_report.json), [Fixture](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/eval/fixture_sealed150_seed12000/report.md), [Baseline](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/eval/pretrained_sealed20_seed12000/report.md), [Telemetry](https://huggingface.co/Rohitpro1/pack-station-training/blob/55b036781ed354634b5f6cc7b562149632b2a8dc/status.json), [Training code](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/scripts/gpu_train_retail.sh), [Sources](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/configs/retail_datasets.yaml), [API](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/API.md), [Checks](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/VERIFICATION.md), [Ops](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/OPERATOR_GUIDE.md)
