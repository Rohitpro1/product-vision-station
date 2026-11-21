# Evaluation report — profile `pretrained`

- Implementation commit: `473b608060c541231f73b3809db2af8b1dca0459`
- Ground truth: `data/sealed/scenes/ground_truth_first20.json` sha256 `c9616e4764c0f161…`
- Scenes: 20 · thresholds `thr-dev-0.5` · catalogue `fixture-r1`
- Backends: `{"detector": "grounding-dino:IDEA-Research/grounding-dino-tiny@a2bb814dd30d", "masks": "sam2:facebook/sam2.1-hiera-tiny@de431c404385", "embedder": "dinov2:facebook/dinov2-small@ed25f3a31f01", "gallery": "dinov2:facebook/dinov2-small@ed25f3a31f01#3ab064159054efcf", "template": "sift-homography-v1", "barcode": "zxing-cpp", "ocr": "unavailable"}`
- Caveat: **synthetic scenes; not real 5 Star presentations**

| Metric | k / n | Value | 95% CI | Gate |
|---|---|---|---|---|
| localisation_recall | 25/33 | 0.7576 | [0.577, 0.889] | fail (>=0.97) |
| exact_sku_precision | 11/11 | 1.0000 | [0.715, 1.000] | pass (>=0.99) |
| exact_sku_recall | 11/21 | 0.5238 | [0.298, 0.743] | fail (>=0.95) |
| exact_sku_recall_identity_observable | 11/13 | 0.8462 | [0.546, 0.981] | fail (>=0.95) |
| false_target_acceptance | 0/12 | 0.0000 | [0.000, 0.265] | insufficient_evidence (point=0.0000, upper95=0.2209 > 0.01; need >=299 clean independent negatives) |
| axis_coverage | 25/33 | 0.7576 | [0.577, 0.889] | fail (>=0.95) |
| axis_p95_err_deg | n=25 | 0.52° | — | pass (<=10.0) |
| heading_coverage | 13/13 | 1.0000 | [0.753, 1.000] | pass (>=0.9) |
| heading_p95_err_deg | n=13 | 0.02° | — | pass (<=15.0) |
| heading_reversal_rate | 0/13 | 0.0000 | [0.000, 0.247] | pass (<0.01) |

Processing time (ms, processing-only): median 12623.175210499994, p95 13305.50874284993

Raw counts: `{"gt": 33, "matched": 25, "sku_accepted": 11, "sku_correct": 11, "target_presentations": 21, "target_correct": 11, "neg_presentations": 12, "neg_false_accept": 0, "axis_obs": 33, "axis_ret": 25, "head_obs": 13, "head_ret": 13, "reversals": 0, "empty_scenes": 3, "empty_false_pos": 0, "target_id_observable": 13, "target_id_observable_correct": 11}`

Negative slices: `{"similar_gold_wrapper": {"false_accept": 0, "n": 12}}`

Failures listed: 13 (see report.json)
