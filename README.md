# Decision Assistant

A production-ready churn prediction and strategy recommendation system for insurance policies.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (<5 minutes)
python run.py
```

All outputs will be generated in the `out/` directory.

## Primary Metric

**AUC-PR (Average Precision)** - See `out/metrics.json`

This metric is optimal for imbalanced classification where we care about precision at the top of the score distribution.

## Key Outputs

### Model Performance
- **`out/metrics.json`** - 7 metrics including AUC-PR (primary), ROC-AUC, Brier score, precision@1%, precision@5%, and prevalence
- **`out/preds_test.csv`** - Test set predictions with policy IDs and probabilities
- **`out/shap_bar.png`** - Top-20 feature importances via SHAP

### Data & Validation
- **`out/data.csv`** - Synthetic panel data (2,000 policies × 12 months)
- **`out/split_report.json`** - Temporal split validation (train/val/test boundaries)
- **`out/leakage_report.txt`** - Leakage trap columns that were excluded

### RAG Strategy Generation
- **`out/lapse_plans.jsonl`** - 3 lapse prevention plans (high/mid/low risk)
- **`out/lead_plans.jsonl`** - 3 lead conversion plans
- **`out/audit_rag.json`** - Faithfulness audit (citations ⊆ retrieved docs)
- **`out/rag/lapse/`** - 6 retention strategy documents
- **`out/rag/lead/`** - 6 conversion strategy documents

### Runtime Metadata
- **`out/run_meta.json`** - Seed, versions, elapsed time, SHAP sample size

## What This Does

1. **Generates Synthetic Data** - Creates realistic insurance panel data with temporal structure, concept drift (starting 2023-07), and a leakage trap feature
2. **Trains Churn Classifier** - Uses XGBoost with temporal split (no data leakage), hyperparameter tuning (20 trials), and early stopping
3. **Explains Predictions** - SHAP analysis identifies key risk drivers
4. **Generates Strategies** - TF-IDF-based RAG produces actionable retention and conversion plans with citations
5. **Validates Quality** - Faithfulness audit ensures all citations come from retrieved documents

## Why These Design Choices

- **Temporal Split** - Train on past data, predict future (2023-01 to 2023-08 train, 2023-09 to 2023-10 val, 2023-11 to 2023-12 test) to prevent leakage
- **Leakage Trap** - Intentionally includes `post_event_call_count` which is dropped; validates that our pipeline catches leakage
- **XGBoost with Manual Search** - Avoids Pipeline+eval_set compatibility issues; 20 trials balance speed and quality
- **AUC-PR Focus** - More appropriate than ROC-AUC for imbalanced data where we care about top predictions
- **SHAP (Global)** - Mean absolute SHAP values reveal which features drive lapse risk across the population
- **TF-IDF RAG** - Fast, offline, deterministic; retrieves 3 docs for 3-step plans with verifiable citations
- **Faithfulness Audit** - Programmatically ensures every citation appears in retrieved documents (target: 100%)

## Next Steps (Not Required by Assignment)

- Calibration plots to assess probability quality
- Threshold tuning for operational deployment
- Cost-sensitive learning to optimize business metrics
- More sophisticated RAG with embeddings or LLMs (currently template-based for speed/determinism)

## Technical Notes

- **Runtime**: <5 minutes on typical laptop
- **Offline**: No API keys, no downloads, no internet required
- **Reproducible**: Seed=42, exact package versions, deterministic algorithms
- **Primary Metric**: AUC-PR (average precision)

See `DISCUSSION.md` for technical deep-dive on leakage trap, drift mechanism, SHAP insights, and ablation study.

