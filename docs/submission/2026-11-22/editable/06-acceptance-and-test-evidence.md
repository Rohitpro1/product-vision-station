# Acceptance and test evidence

Public retail crops, synthetic scenes, hosted requests and a physical installation answer different questions. These results cannot be combined into an overall accuracy percentage.

| Evidence and gate | Measured result | Disposition |
| --- | --- | --- |
| Retail • package recall ≥97% at IoU 0.5 | 347/360 = 96.39%; 95% interval 93.90–98.06%. Precision 347/359 = 96.66%. | Public-data point estimate below target. No field acceptance. |
| Retail • exact SKU recall ≥95%; precision ≥99% | Ground-truth crops: 0/3 correct targets; 0/7 correct accepted SKUs. Family or better 1/3. Detector located 3/3. | Two Classic bars returned unknown; the third was called 3D. |
| Retail • false accepts ≤1% | 6/357 = 1.68%; one-sided upper 95% bound 3.29%. Correlated boxes limit interval interpretation. | Fails the target on this sample. |
| Sealed fixture • 150 scenes | Scripted localisation. SKU precision 103/103; recall 103/145 overall, 103/105 on observable views. Negatives 0/83. | Overall recall below target. Upper bound 3.54% on false accepts. |
| Fixture • geometry | Axis p95 0.15° (228 returns); heading p95 0.05° (105). Pretrained baseline found 25/33 boxes over 20 synthetic scenes. | Demonstrates geometry with fixtures. Real-wrapper angular accuracy remains unmeasured. |
| Physical joint yield / timing | Target ≥85% correct SKU + heading within 15° and 500 ms on eligible presentations. Target 15 fps, p95 ≤150 ms, 8 h. | Open. T4 detection evaluation 34.2 ms/image and hosted still timings measure different boundaries. |

## Software, release and service checks

GitHub Actions run 45037028660 for b56c2a7 succeeded on deterministic tests, training smoke and Postgres migrations. The opt-in real-model job was skipped. The handoff separately reports 169 passes and two skips at ed64b5e. This review checked the CI record, live metadata and binary hashes; the station hardware benchmark was not repeated.

Recorded browser and real-photo tests produced a package box and an axis near 4.6–4.7° on both backends; real product identity was unknown. A same-photo rotation after enrolment demonstrates matching, but cannot establish new-pack recognition.

Sources: [CI](https://github.com/Rohitpro1/product-vision-station/actions/runs/45037028660), [Checks](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/VERIFICATION.md), [V1 evaluation](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/training/real5star_kaggle_t4x2/eval_report.json), [Fixture](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/eval/fixture_sealed150_seed12000/report.md), [Baseline](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/eval/pretrained_sealed20_seed12000/report.md), [Requirements](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/REQUIREMENTS_EVIDENCE.md)
