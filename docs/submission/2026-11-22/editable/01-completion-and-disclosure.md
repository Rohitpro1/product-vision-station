# Completion and disclosure

Owner: Rohit Kumar P. Scope: one RGB product-inspection station and a public still-image demonstration. Evidence review dated 22 November 2026; training telemetry frozen at 22:07 IST.

**DELIVERY DISPOSITION** The software and hosted package-detection demo are delivered. Product-level acceptance remains open. A detected box, a named SKU and a directed angle are three independently evaluated outputs.

| Deliverable | Observed state | Acceptance boundary |
| --- | --- | --- |
| Local station | CLI, file/replay/camera capture, tracking, fusion, preview, metrics, bundle and event tooling exist in main. | Camera installation, hardware timing and eight-hour operation require field evidence. |
| Hosted inspection | The frontend selects one of two HTTP services. Both returned ready, with RT-DETR on CPU and fixture-r1. | Trained package boxes and mask-derived axes. The hosted catalogue contains two synthetic Zephyr SKUs. |
| Run 1 • completed | Best of 40 epochs exported to ONNX. Test set contains 350 photos from two Roboflow exports. | 347/360 package boxes found; all three 5 Star boxes located. Correct exact 5 Star identities: 0/3. |
| Retail v2 • candidate | Prepared data and a complete smoke pipeline; full run reached epoch 10/20 at the evidence cutoff. | No final per-brand or end-to-end qualification was available in the inspected training revision. |

## Basis for submission

Design intent comes from the original architecture and build brief. The implementation baseline is ed64b5e; main revision b56c2a7 adds the handoff. Release claims below use the deployed metadata, committed reports, model bytes and CI record. A readiness check establishes a service response at review time. Historical upload timings and station benchmarks carry their own provenance.

The dossier includes the deployed and local architectures, dataset and model records, acceptance results, operating limits, deployment commands and release conditions. The companion archive includes source registers and machine-readable evidence.

Sources: [Design basis](https://chatgpt.com/library?file_id=libfile_bd77376e25f48191bf074039b4286799&account_id=personal), [Build brief](https://chatgpt.com/library?file_id=libfile_cbff2f8433708191ae98351160388505&account_id=personal), [Main](https://github.com/Rohitpro1/product-vision-station/tree/b56c2a781939fc1524a5b9d2e1ba297992d58bfb), [Handoff](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/HANDOFF.md), [HF meta](https://rohitpro1-pack-station-api.hf.space/v1/meta), [Vercel meta](https://pack-station-api.vercel.app/v1/meta), [V1 evaluation](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/evidence/training/real5star_kaggle_t4x2/eval_report.json), [Telemetry](https://huggingface.co/Rohitpro1/pack-station-training/blob/55b036781ed354634b5f6cc7b562149632b2a8dc/status.json)
