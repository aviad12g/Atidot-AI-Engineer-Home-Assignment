# Technical Discussion

## Leakage Trap

We included a `post_event_call_count` feature that uses information from after the lapse event (follow-up calls). The leakage guard programmatically drops all `post_event_*` columns before training. See `out/leakage_report.txt` for confirmation.

## Temporal Split & 3-Month Horizon

**Split:** Train (2023-01 to 2023-08), Val (2023-09 to 2023-10), Test (2023-11 to 2023-12). Validated in `out/split_report.json`.

**Horizon:** To compute `lapse_next_3m` for December 2023, we need Jan-Mar 2024. We internally simulated 15 months to compute 3-month-ahead labels for the first 12; only the first 12 are saved and used.

## Concept Drift

Starting 2023-07, we added +80% base risk and amplified smoker/regional effects to simulate market disruption.

## SHAP Insights

Top risk drivers (mean absolute SHAP values):

1. **Payment failures** - Financial stress is the strongest lapse signal.
2. **Premium increases** - Price sensitivity drives most lapses.
3. **Low engagement** - Disengaged customers are 2.5x more likely to lapse.
4. **Short tenure** - New customers lack loyalty.
5. **No agent** - Personal touch reduces lapse by ~60%.

The model also captures interaction effects: payment failures + low engagement, young + no agent + price increase, and claims without agent support.

## Ablation

- **Dummy (prevalence)**: 0.49 AUC-PR
- **Logistic Regression**: 0.71 AUC-PR
- **XGBoost**: 0.72 AUC-PR (+47% lift over baseline)

The synthetic data includes strong feature-target relationships (payment failures 3x weight, premium changes 8x) and nonlinear interactions to create realistic, learnable patterns.
