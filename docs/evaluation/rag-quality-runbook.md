# RAG Quality Release Runbook

Run these commands in order for a public-v1 release candidate:

```powershell
python -m pytest -q
python scripts/migrate_db.py
python scripts/validate_eval_dataset.py --dataset evaluation/data/release_qa.v1.jsonl --require-release-quota
python scripts/run_release_eval.py --dataset evaluation/data/release_qa.v1.jsonl --split development --repeat 1 --output evaluation/reports/development.json
$developmentRunId = (Get-Content evaluation/reports/development.json -Raw | ConvertFrom-Json).run_id
python scripts/calibrate_local_judge.py --run-id $developmentRunId --output evaluation/reports/judge-calibration.json
python scripts/run_release_eval.py --dataset evaluation/data/release_qa.v1.jsonl --split holdout --repeat 3 --output-dir evaluation/reports/holdout
python scripts/summarize_beta.py --days 14 --output evaluation/reports/beta-public-v1.json
```

Holdout output cannot be used to tune prompts, evidence thresholds, retriever settings, or judge prompts for the same release candidate. If holdout fails, return to the development split, create a new frozen release configuration, and run a new release candidate.

Release remains blocked until the dataset has exact release quotas, local judge calibration is eligible or generation scores are human-reviewed, two consecutive holdout runs pass, beta gates pass, and all critical leakage/error counts are zero.
