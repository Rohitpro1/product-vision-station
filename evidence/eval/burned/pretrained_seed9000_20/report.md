# Evaluation report — profile `pretrained`

- Implementation commit: `None`
- Ground truth: `data/heldout/scenes/ground_truth_first20.json` sha256 `0ad2eb9f34d7a18d…`
- Scenes: 20 · thresholds `thr-dev-0.4` · catalogue `fixture-r1`
- Backends: `{"detector": "grounding-dino:IDEA-Research/grounding-dino-tiny@a2bb814dd30d", "masks": "sam2:facebook/sam2.1-hiera-tiny@de431c404385", "embedder": "dinov2:facebook/dinov2-small@ed25f3a31f01", "gallery": "dinov2:facebook/dinov2-small@ed25f3a31f01#3ab064159054efcf", "template": "sift-homography-v1", "barcode": "zxing-cpp", "ocr": "unavailable"}`
- Caveat: **synthetic scenes; not real 5 Star presentations**

| Metric | k / n | Value | 95% CI | Gate |
|---|---|---|---|---|
| localisation_recall | 21/24 | 0.8750 | [0.676, 0.973] | fail (>=0.97) |
| exact_sku_precision | 10/10 | 1.0000 | [0.692, 1.000] | pass (>=0.99) |
| exact_sku_recall | 10/17 | 0.5882 | [0.329, 0.816] | fail (>=0.95) |
| exact_sku_recall_identity_observable | 10/14 | 0.7143 | [0.419, 0.916] | fail (>=0.95) |
| false_target_acceptance | 0/7 | 0.0000 | [0.000, 0.410] | insufficient_evidence (point=0.0000, upper95=0.3482 > 0.01; need >=299 clean independent negatives) |
| axis_coverage | 21/24 | 0.8750 | [0.676, 0.973] | fail (>=0.95) |
| axis_p95_err_deg | n=21 | 0.91° | — | pass (<=10.0) |
| heading_coverage | 14/14 | 1.0000 | [0.768, 1.000] | pass (>=0.9) |
| heading_p95_err_deg | n=14 | 0.05° | — | pass (<=15.0) |
| heading_reversal_rate | 0/14 | 0.0000 | [0.000, 0.232] | pass (<0.01) |

Processing time (ms, processing-only): median 13053.04267849999, p95 17046.584865550056

Raw counts: `{"gt": 24, "matched": 21, "sku_accepted": 10, "sku_correct": 10, "target_presentations": 17, "target_correct": 10, "neg_presentations": 7, "neg_false_accept": 0, "axis_obs": 24, "axis_ret": 21, "head_obs": 14, "head_ret": 14, "reversals": 0, "empty_scenes": 7, "empty_false_pos": 0, "target_id_observable": 14, "target_id_observable_correct": 10}`

Negative slices: `{"similar_gold_wrapper": {"false_accept": 0, "n": 7}}`

Failures listed: 8 (see report.json)
