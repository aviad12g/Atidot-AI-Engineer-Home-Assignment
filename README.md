# Insurance Decision Assistant

ML pipeline for insurance churn prediction with RAG-powered retention strategies.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Runtime: ~21 seconds. All outputs go to `out/`.

## What It Does

- Trains XGBoost on 2,000 policies x 12 months with temporal split (train/val/test by month)
- Predicts 3-month lapse probability; evaluates with AUC-PR, precision@k
- Generates 3 lapse-prevention plans (high/mid/low risk) + 3 lead-conversion plans via TF-IDF RAG

## Outputs

- `metrics.json` - AUC-PR, precision@1%, precision@5%, prevalence
- `preds_test.csv` - Test predictions with probabilities
- `model.pkl` - Trained XGBoost pipeline
- `shap_bar.png` - Top-15 feature importances
- `lapse_plans.jsonl` / `lead_plans.jsonl` - RAG strategies with citations
- `audit_rag.json` - Faithfulness verification (100%)

## Results

**AUC-PR: 0.72** | Precision@1%: 0.80 | Precision@5%: 0.83

Top risk drivers (SHAP): payment failures, premium increases, low engagement, short tenure, no agent.

---

See `DISCUSSION.md` for technical notes on leakage trap, temporal split, and SHAP insights.
