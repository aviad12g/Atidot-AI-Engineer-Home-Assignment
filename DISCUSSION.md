# Technical Discussion

## Leakage Trap

We included a `post_event_call_count` feature that uses information from after the lapse event (follow-up calls). The leakage guard programmatically drops all `post_event_*` columns before training. See `out/leakage_report.txt` for confirmation.

## Temporal Split & 3-Month Horizon

**Split:** Train (2023-01 to 2023-08), Val (2023-09 to 2023-10), Test (2023-11 to 2023-12). Validated in `out/split_report.json`.

**Horizon:** To compute `lapse_next_3m` for December 2023, we need Jan-Mar 2024. We internally simulated 15 months to compute 3-month-ahead labels for the first 12; only the first 12 are saved and used.

## Concept Drift

Starting 2023-07, we added +80% base risk and amplified smoker/regional effects to simulate market disruption.

## SHAP Insights

Top risk drivers (mean absolute SHAP values, 300 test samples):

1. **No agent relationship** – Lack of an agent remains the largest single driver of lapse risk.
2. **Age curve** – Very young and older policyholders push predictions upward, consistent with the synthetic risk design.
3. **Smoker status** – Smoker vs. non-smoker signals materially shift risk scores.
4. **Regional exposure (South/West)** – These regions contribute the strongest uplift after the drift event.
5. **Engagement & dependents depth** – Lower engagement and fewer dependents raise risk, while high engagement offsets it.

Payment failures and recent claims are now part of the feature set; they surface mainly through interaction splits rather than the global top-5 importances.

## Ablation

- **Dummy (prevalence)**: 0.49 AUC-PR
- **Logistic Regression**: 0.71 AUC-PR
- **XGBoost**: 0.72 AUC-PR (+47% lift over baseline)

The synthetic data includes strong feature-target relationships (payment failures 3x weight, premium changes 8x) and nonlinear interactions to create realistic, learnable patterns.
