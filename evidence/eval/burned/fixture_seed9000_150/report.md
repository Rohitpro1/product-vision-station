# Evaluation report — profile `fixture`

- Implementation commit: `None`
- Ground truth: `data/heldout/scenes/ground_truth.json` sha256 `ee6127a7e581cf70…`
- Scenes: 150 · thresholds `thr-dev-0.4` · catalogue `fixture-r1`
- Backends: `{"detector": "scripted-fixture", "masks": "scripted-fixture", "embedder": "colour-hist-v1", "gallery": "colour-hist-v1#7b16c94878feebca", "template": "sift-homography-v1", "barcode": "zxing-cpp", "ocr": "unavailable"}`
- Caveat: **fixture profile uses scripted detections; only identity/orientation/barcode/fusion algorithms are exercised**

| Metric | k / n | Value | 95% CI | Gate |
|---|---|---|---|---|
| localisation_recall | 219/219 | 1.0000 | [0.983, 1.000] | pass (>=0.97) |
| exact_sku_precision | 111/111 | 1.0000 | [0.967, 1.000] | pass (>=0.99) |
| exact_sku_recall | 111/146 | 0.7603 | [0.683, 0.827] | fail (>=0.95) |
| exact_sku_recall_identity_observable | 111/111 | 1.0000 | [0.967, 1.000] | pass (>=0.95) |
| false_target_acceptance | 0/73 | 0.0000 | [0.000, 0.049] | insufficient_evidence (point=0.0000, upper95=0.0402 > 0.01; need >=299 clean independent negatives) |
| axis_coverage | 219/219 | 1.0000 | [0.983, 1.000] | pass (>=0.95) |
| axis_p95_err_deg | n=219 | 0.15° | — | pass (<=10.0) |
| heading_coverage | 111/111 | 1.0000 | [0.967, 1.000] | pass (>=0.9) |
| heading_p95_err_deg | n=111 | 0.05° | — | pass (<=15.0) |
| heading_reversal_rate | 0/111 | 0.0000 | [0.000, 0.033] | pass (<0.01) |

Processing time (ms, processing-only): median 36.75777399996605, p95 107.59348499994984

Raw counts: `{"gt": 219, "matched": 219, "sku_accepted": 111, "sku_correct": 111, "target_presentations": 146, "target_correct": 111, "neg_presentations": 73, "neg_false_accept": 0, "axis_obs": 219, "axis_ret": 219, "head_obs": 111, "head_ret": 111, "reversals": 0, "empty_scenes": 39, "empty_false_pos": 0, "target_id_observable": 111, "target_id_observable_correct": 111}`

Negative slices: `{"similar_gold_wrapper": {"false_accept": 0, "n": 73}}`

Failures listed: 35 (see report.json)
