# Documentation index

| Document | Read it for |
|---|---|
| [`submission/`](submission/) | **Engineering submission packs**, frozen and checksummed. The latest is [`submission/2026-11-22/`](submission/2026-11-22/), whose entry point is its [completion report PDF](submission/2026-11-22/Product_Vision_Completion_Report.pdf). |
| [`HANDOFF.md`](HANDOFF.md) | The full project context for a new engineer or agent: requests, the live system, both training runs, datasets, diagrams, scripts, hosting, fixes and open items. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Runtime pipeline, identity rules, outbox, data-to-release flow and execution profiles (derived from ADR-01). |
| [`API.md`](API.md) | REST and WebSocket API, FrameResult schema 1.0. |
| [`OPERATOR_GUIDE.md`](OPERATOR_GUIDE.md) | Install, enrolment, bundles, rollback and station operation. |
| [`TRAINING_EVALUATION.md`](TRAINING_EVALUATION.md) | Training, export, evaluation, and the GPU runbooks (run 1 and retail v2 on Kaggle). |
| [`VERIFICATION.md`](VERIFICATION.md) | What was verified, with raw counts and intervals, the hosted checks, and what is not run. |
| [`REQUIREMENTS_EVIDENCE.md`](REQUIREMENTS_EVIDENCE.md) | Requirement → evidence map. |
| [`DECISIONS.md`](DECISIONS.md) | Decision log (D1 onwards) and model-selection status. |
| [`EXTENSIONS.md`](EXTENSIONS.md) | Planned extensions. |
| [`reference_drawing.md`](reference_drawing.md) | Reference drawing conventions (axis, heading, faces). |

Where things disagree, trust them in this order:

1. The **live state**: the services' `/readyz` and `/v1/meta`, and the HF training repo.
2. The **repository documents** at `main`.
3. The **dated submission packs**, which are deliberately frozen snapshots.
