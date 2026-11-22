# Product Vision Station • submission index

**Owner**: Rohit Kumar P. **Evidence cutoff**: 22 Nov 2026, retail v2 status at 22:07 IST. **Disposition**: v1 software and hosted demonstrator delivered; real-product identity and field commissioning open.

Start with Product_Vision_Completion_Report.pdf. The documents/ directory contains self-contained extracts with their original dossier page numbers. The architecture file includes the hosted system, station processing and candidate training lifecycle. Vector originals are in diagrams/ (DOT, SVG and PDF). Markdown equivalents are in editable/.

## Machine-readable records

- registers/dataset_register.csv contains two version-confirmed v1 dataset rows and 38 v2 candidate rows. The URL and cap are taken from the configuration; historical per-source use in v2 requires both its download report and the list of projects with retained images. Blank caps indicate no source-specific configured value.
- registers/model_register.csv links upstream models, revisions and checksums. Runtime and training pins are tracked in requirements/ in Git.
- registers/code_register.csv indexes modules, tests, scripts, workflow, config, frontend, migration and deployment paths at the audited revision. Symbols describe the source tree and may be inactive in a given execution profile.
- registers/acceptance_register.csv gives populations and raw counts. registers/source_register.csv links the supporting primary sources.
- evidence/ contains the committed run-1 reports, ground-truth crop metrics, dataset statistics, API schema, fixture reports, current-as-reviewed host responses, CI job metadata and the frozen v2 training state. Evidence with “fixture” in its name measures synthetic cases. The runnable source and model archives are linked via the source register; this documentation pack contains no raw training images, credentials or checkpoints.
- SHA256SUMS.txt covers every other file in this package. To verify from the extracted root on Linux: sha256sum -c SHA256SUMS.txt.

## Update after v2

Preserve this snapshot. For the next revision, record the HF result commit, join download_report.json (accepted versions, licences and skips) to data_stats.json (projects_used, images_by_project, splits and leakage), compare the held-out detector, per-brand crop and full-image E2E reports, and verify served /v1/meta on both production aliases before declaring a model deployed. Training progress alone cannot establish a production switch.
