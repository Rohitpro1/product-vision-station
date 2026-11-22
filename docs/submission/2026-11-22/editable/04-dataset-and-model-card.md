# Dataset and model card

The completed experiment uses two named versions. Retail v2 selects project versions at download time, then filters them. The register supplied with this report records each configured source and its evidence state.

| Run 1 source / unit | Recorded provenance |
| --- | --- |
| SCt v6 | Roboflow Universe • nano-b8y2n/sct-ufzbr • COCO detection export. |
| Scancart v2 | Roboflow Universe • nano-b8y2n/scancart • COCO detection export. Both releases are identified as CC BY 4.0 in the training evidence. |
| Merged images | 2,729 images; 1,533 perceptual groups; 2,291 images in duplicate groups. No detected group overlap. The available labels include 151 original names and spelling variants. |
| Label transformation | All annotated products train a single generic package class. Separate reference crops retain product identity. Source boxes provide no physical A/B endpoints. |

| Split | Images | Other boxes | 5 Star Classic | 3D | Bites |
| --- | --- | --- | --- | --- | --- |
| Train | 1,893 | 1,987 | 21 | 4 | 2 |
| Validation | 486 | 495 | 0 | 0 | 0 |
| Test | 350 | 357 | 3 | 0 | 0 |

## Training and export

COCO-initialised PekingU/rtdetr_r18vd was fine-tuned at 640 px with PyTorch 2.6.0+cu124, Python 3.13.16 and two Tesla T4s. DDP split a global batch of 32 across GPUs. Learning rate 0.000283; 40 epochs in 42.5 minutes. Validation recall plus precision selected epoch 11. ONNX maximum absolute difference from the reference was 1.00 × 10⁻⁵.

Offline identity evaluation used a frozen DINOv2-small gallery of 1,772 reference vectors, including 27 target boxes. Validation chose cosine 0.40 and margin 0.01 with no target-positive validation example. The hosted demo ships the detector and synthetic catalogue. Stock-image watermarks, shared author and repeated pack imagery constrain generalisation.

Alternative pretrained profile: Grounding DINO proposals, SAM 2.1 masks and DINOv2 embeddings. A separate landmark/face head has a training and export smoke test. Its checkpoint has no real-product qualification. Source revisions, file hashes and licences appear in the model manifest.

Sources: [SCT](https://universe.roboflow.com/nano-b8y2n/sct-ufzbr/dataset/6), [Scancart](https://universe.roboflow.com/nano-b8y2n/scancart/dataset/2), [V1 training](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/training/real5star_kaggle_t4x2/train_report.json), [V1 evaluation](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/training/real5star_kaggle_t4x2/eval_report.json), [Models](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/models/manifest.json), [Notices](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/THIRD_PARTY_NOTICES.md)
