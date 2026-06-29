# NomNom RAG Evaluation

This folder stores offline RAG evaluation datasets, source inventories, baselines, and generated reports.

Use `scripts/run_release_eval.py` for release decisions. It runs the production RAG path, disables response cache, records decisions/provenance, and evaluates against `evaluation/baselines/nomnom_public_v1.json`.

```powershell
python scripts/validate_eval_dataset.py --dataset evaluation/data/release_qa.v1.jsonl
python scripts/run_release_eval.py --dataset evaluation/data/release_qa.v1.jsonl --split development --repeat 1 --output evaluation/reports/development.json
python scripts/run_release_eval.py --dataset evaluation/data/release_qa.v1.jsonl --split holdout --repeat 3 --output-dir evaluation/reports/holdout
```

`scripts/run_ragas_eval.py` is only for optional experiments. It is not the release command.

Release gates are defined in `evaluation/baselines/nomnom_public_v1.json` and fail closed when a required metric is missing. `product_ready` remains false until dataset quota, holdout repeats, local judge calibration, and beta gates are all backed by reviewed data.
