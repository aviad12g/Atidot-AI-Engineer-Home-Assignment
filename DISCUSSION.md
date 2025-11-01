# Technical Discussion

## Leakage Trap

We included a `post_event_call_count` feature that uses information from after the lapse event (follow-up calls). The leakage guard programmatically drops all `post_event_*` columns before training. See `out/leakage_report.txt` for confirmation.

## Temporal Split & 3-Month Horizon

Train: 2023-01 to 2023-08 (8 months), Val: 2023-09 to 2023-10 (2 months), Test: 2023-11 to 2023-12 (2 months). Prediction month boundaries are validated in `out/split_report.json`.

To compute the `lapse_next_3m` target for December 2023, we need to observe Jan-Mar 2024. We internally simulated 15 months (2023-01 through 2024-03) to compute 3-month-ahead labels for the first 12; only the first 12 are saved and used.

## Concept Drift

Starting in 2023-07, we added +80% base risk and amplified smoker/regional effects to simulate market disruption. The temporal split ensures the model learns from drift-period data in training and is tested on fully drifted months.

## SHAP Insights

Global feature importance (mean absolute SHAP values) reveals:

1. **Payment failures** - Most predictive. Financial stress is the strongest lapse signal.
2. **Premium increases** - Price sensitivity drives most lapses (8x weight in data generation).
3. **Low engagement** - Disengaged customers are 2.5x more likely to lapse.
4. **Short tenure** - New customers haven't built loyalty yet.
5. **No agent** - Personal touch reduces lapse probability by ~60%.

The model also captures interaction effects: payment failures + low engagement (disengaged at-risk), young + no agent + price increase (triple-risk scenario), and claims without agent support (unresolved dissatisfaction).

## Ablation Study

We trained three models on the same temporal split:

- **Dummy (prevalence)**: 0.49 AUC-PR (baseline)
- **Logistic Regression**: 0.71 AUC-PR
- **XGBoost**: 0.72 AUC-PR (+47% lift over baseline)

The XGBoost model achieves strong performance (0.60-0.72 is typical for real-world insurance churn) due to:
- Strong feature-target relationships (payment failures 3x weight, premium changes 8x)
- Clear risk segments (young + no agent, smokers + premium increase)
- Nonlinear interactions (price shock + low tenure, claims + no agent)
- Behavioral signals (engagement, claims, payment history)

This synthetic data was carefully engineered to create realistic, learnable patterns without being trivially separable.
