# Extension boundaries (build prompt section 11; PDF p.3, p.10)

All of these are **disabled** in the base station. Each needs its own evidence and approval.

| Extension | Interface / design | Trigger condition |
|---|---|---|
| Tabletop-frame angle | `calibration.homography` (3×3 image→table) in station YAML; transform **both** endpoints A and B, then compute the angle in table coordinates; reject when the mask/endpoint residual implies out-of-plane pose | the application needs a world/table angle; requires a verified intrinsic + plane calibration (AprilTag board) |
| Metric 3D pose | separate optional module: rigid object model with defined frame, calibrated camera, independently measured ground truth, IPPE/solvePnP candidates with ambiguity rejection, BOP-style symmetry-aware metrics | rigid packaging only; a flexible wrapper needs a deformation model |
| Second view | second synchronised camera for hidden distinguishing text | fronts of SKUs are identical at station resolution |
| Handheld scanner | explicit one-item scan session binds GTIN to the single active track | one-item-at-a-time workflows |
| RFID | EPC→SKU mapping; only with reliable item-to-image association | tagged stock; multi-item reads need spatial association |
| Phone IMU | measures the *phone*; combine time-aligned camera-to-world with a measured object-to-camera pose | never as a substitute for object orientation |

FoundationPose (NVIDIA licence: non-commercial research/evaluation) stays outside any commercial bundle unless
rights are documented.
