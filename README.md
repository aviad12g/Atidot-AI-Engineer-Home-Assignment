# Insurance Decision Assistant

This repo holds a compact churn-style pipeline that doubles as a retention assistant. One command (`python run.py`) stands up synthetic data, trains a model, and drafts action plans.

## How to Run It

```bash
pip install -r requirements.txt
python run.py
```

The run finishes in under a minute on a typical laptop and writes every artifact to `out/`.

## Inside the Pipeline

- Builds 24k synthetic monthly rows (2k policies × 12 months) with drift and a hidden leakage trap.
- Trains an XGBoost model with strict temporal splits and lightweight tuning.
- Evaluates with AUC-PR plus precision@1% and precision@5%.
- Generates three lapse-prevention playbooks (high / mid / low risk) and three lead-conversion plans using TF‑IDF retrieval on small markdown corpora.

## Key Artifacts

- `metrics.json` – model metrics on the hold-out set.
- `preds_test.csv` – month-level predictions with probabilities.
- `model.pkl` – preprocessor + tuned XGBoost bundle.
- `shap_bar.png` – top feature importances from 300 sampled test rows.
- `lapse_plans.jsonl`, `lead_plans.jsonl` – generated plans with `[Doc#]` citations.
- `audit_rag.json` – simple faithfulness check (should read 100%).

## Current Model Snapshot

- AUC-PR: 0.718
- Precision@1%: 0.775
- Precision@5%: 0.825
- Runtime: under 1 minute

The strongest global signals in SHAP are lack of an agent relationship, age extremes, smoker status, and post-drift regional effects. More color on data generation, leakage handling, and interpretability lives in `DISCUSSION.md`.
