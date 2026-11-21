# Evaluation report — profile `fixture`

- Implementation commit: `473b608060c541231f73b3809db2af8b1dca0459`
- Ground truth: `data/sealed/scenes/ground_truth.json` sha256 `893b1140909a495c…`
- Scenes: 150 · thresholds `thr-dev-0.5` · catalogue `fixture-r1`
- Backends: `{"detector": "scripted-fixture", "masks": "scripted-fixture", "embedder": "colour-hist-v1", "gallery": "colour-hist-v1#7b16c94878feebca", "template": "sift-homography-v1", "barcode": "zxing-cpp", "ocr": "unavailable"}`
- Caveat: **fixture profile uses scripted detections; only identity/orientation/barcode/fusion algorithms are exercised**

| Metric | k / n | Value | 95% CI | Gate |
|---|---|---|---|---|
| localisation_recall | 228/228 | 1.0000 | [0.984, 1.000] | pass (>=0.97) |
| exact_sku_precision | 103/103 | 1.0000 | [0.965, 1.000] | pass (>=0.99) |
| exact_sku_recall | 103/145 | 0.7103 | [0.629, 0.783] | fail (>=0.95) |
| exact_sku_recall_identity_observable | 103/105 | 0.9810 | [0.933, 0.998] | pass (>=0.95) |
| false_target_acceptance | 0/83 | 0.0000 | [0.000, 0.043] | insufficient_evidence (point=0.0000, upper95=0.0354 > 0.01; need >=299 clean independent negatives) |
| axis_coverage | 228/228 | 1.0000 | [0.984, 1.000] | pass (>=0.95) |
| axis_p95_err_deg | n=228 | 0.15° | — | pass (<=10.0) |
| heading_coverage | 105/105 | 1.0000 | [0.965, 1.000] | pass (>=0.9) |
| heading_p95_err_deg | n=105 | 0.05° | — | pass (<=15.0) |
| heading_reversal_rate | 0/105 | 0.0000 | [0.000, 0.035] | pass (<0.01) |

Processing time (ms, processing-only): median 31.41128350011968, p95 78.42002905010759

Raw counts: `{"gt": 228, "matched": 228, "sku_accepted": 103, "sku_correct": 103, "target_presentations": 145, "target_correct": 103, "neg_presentations": 83, "neg_false_accept": 0, "axis_obs": 228, "axis_ret": 228, "head_obs": 105, "head_ret": 105, "reversals": 0, "empty_scenes": 37, "empty_false_pos": 0, "target_id_observable": 105, "target_id_observable_correct": 103}`

Negative slices: `{"similar_gold_wrapper": {"false_accept": 0, "n": 83}}`

Failures listed: 42 (see report.json)
