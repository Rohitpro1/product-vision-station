# Engineering submissions

Each dated folder is an engineering submission pack, committed **exactly as delivered** so that its own
`SHA256SUMS.txt` still verifies. Don't edit files inside a dated folder. A later revision goes in a new dated folder,
and the pack's README explains how to bring it up to date ("Update after v2").

| Pack | Evidence cutoff | Disposition | Start here |
|---|---|---|---|
| [`2026-11-22/`](2026-11-22/) | 22 Nov 2026, retail v2 status at 22:07 IST (full run at epoch 10 of 20) | v1 software and hosted package-detection demonstrator delivered. Real-product identity and field commissioning are still open. | [`Product_Vision_Completion_Report.pdf`](2026-11-22/Product_Vision_Completion_Report.pdf) |

## Contents of `2026-11-22/`

| Path | What it is |
|---|---|
| [`Product_Vision_Completion_Report.pdf`](2026-11-22/Product_Vision_Completion_Report.pdf) | Full 10-page dossier: delivery decision, architecture, data, verification, deployment, release conditions. |
| [`README.md`](2026-11-22/README.md) | Pack index, how to read the machine-readable records, and how to update after v2. |
| `documents/` | Self-contained PDF extracts that keep their original dossier page numbers: [completion and disclosure](2026-11-22/documents/01_Completion_Disclosure.pdf), [as-built architecture](2026-11-22/documents/Product_Vision_As_Built_Architecture.pdf), [dataset and model card](2026-11-22/documents/03_Dataset_and_Model_Card.pdf), [verification and acceptance](2026-11-22/documents/04_Verification_and_Acceptance.pdf), [deployment and code handover](2026-11-22/documents/05_Deployment_and_Code_Handover.pdf). |
| `editable/` | Markdown equivalents of every section (01 to 10), with diagrams embedded from `../diagrams/*.svg`. |
| `diagrams/` | Vector originals in DOT, SVG and PDF: [hosted v1](2026-11-22/diagrams/01-hosted-v1.svg), [station runtime](2026-11-22/diagrams/02-station-runtime.svg), [training and release](2026-11-22/diagrams/03-training-release.svg). |
| `registers/` | CSV registers: datasets (2 v1 rows and 38 v2 candidate rows), models (revisions and checksums), code (modules, tests, scripts and config at the audited revision), acceptance (populations and raw counts), sources. |
| `evidence/` | Frozen observations: host `/readyz` and `/v1/meta` responses, HF Space and training-repo metadata, CI job metadata, retail v2 status snapshot, and copies of the committed run 1 reports, API schema, fixture reports and model manifest under `evidence/source/`. |
| `SHA256SUMS.txt` | SHA-256 of every other file in the pack. |

Verify the pack from the repository root:

```bash
cd docs/submission/2026-11-22 && sha256sum -c SHA256SUMS.txt
```

The pack contains no raw training images, credentials or checkpoints, and its PDFs embed no raster images.
Its `evidence/source/` files are point-in-time copies of files that still live in the repository, under
`evidence/`, `configs/` and `models/`. When they differ, the copies in the pack describe the submission, and the
repository files describe the current state.
