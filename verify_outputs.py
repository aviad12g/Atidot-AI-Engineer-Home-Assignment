import json
import pandas as pd
import os

print("="*60)
print("COMPREHENSIVE VERIFICATION")
print("="*60)

# 1. Check all required files exist
required_files = [
    'out/data.csv',
    'out/metrics.json',
    'out/preds_test.csv',
    'out/split_report.json',
    'out/leakage_report.txt',
    'out/shap_bar.png',
    'out/lapse_plans.jsonl',
    'out/lead_plans.jsonl',
    'out/audit_rag.json',
    'out/run_meta.json',
]

print("\n[1] File Existence Check:")
all_exist = True
for f in required_files:
    exists = os.path.exists(f)
    print(f"  {'✓' if exists else 'X'} {f}")
    all_exist = all_exist and exists

# 2. Check data.csv structure
print("\n[2] Dataset Verification:")
df = pd.read_csv('out/data.csv')
print(f"  Rows: {len(df)} (expected: 24,000)")
print(f"  Unique policies: {df['policy_id'].nunique()} (expected: 2,000)")
print(f"  Months: {df['month'].nunique()} (expected: 12)")
print(f"  Lapse rate: {df['lapse_next_3m'].mean():.3f}")

required_cols = ['policy_id', 'month', 'age', 'tenure_m', 'premium', 
                 'coverage', 'region', 'has_agent', 'is_smoker', 
                 'dependents', 'lapse_next_3m', 'premium_change_90d']
missing = [c for c in required_cols if c not in df.columns]
print(f"  All required columns: {'YES' if not missing else 'NO - Missing: ' + str(missing)}")

# 3. Check metrics
print("\n[3] Metrics Verification:")
with open('out/metrics.json') as f:
    metrics = json.load(f)
required_metrics = ['auc_pr', 'roc_auc', 'brier', 'precision_at_1pct', 
                   'precision_at_5pct', 'prevalence_train', 'prevalence_test']
for m in required_metrics:
    if m in metrics:
        print(f"  ✓ {m}: {metrics[m]:.4f}")
    else:
        print(f"  X {m}: MISSING")

# 4. Check RAG outputs
print("\n[4] RAG Plans Verification:")
with open('out/lapse_plans.jsonl') as f:
    lapse_plans = [json.loads(line) for line in f]
print(f"  Lapse plans: {len(lapse_plans)} (expected: 3)")
for i, plan in enumerate(lapse_plans):
    print(f"    Plan {i+1}: risk={plan['risk_bucket']}, prob={plan.get('lapse_probability', 0):.3f}, steps={len(plan['plan_steps'])}")

with open('out/lead_plans.jsonl') as f:
    lead_plans = [json.loads(line) for line in f]
print(f"  Lead plans: {len(lead_plans)} (expected: 3)")
for i, plan in enumerate(lead_plans):
    print(f"    Plan {i+1}: steps={len(plan['plan_steps'])}")

# 5. Check faithfulness
print("\n[5] RAG Faithfulness Audit:")
with open('out/audit_rag.json') as f:
    audit = json.load(f)
print(f"  Faithfulness: {audit['faithful_percent']}% (expected: 100%)")
print(f"  Total plans audited: {len(audit['plans'])}")

# 6. Check leakage report
print("\n[6] Leakage Guard:")
with open('out/leakage_report.txt') as f:
    content = f.read()
    has_post_event = 'post_event_call_count' in content
print(f"  Leakage trap detected: {'YES' if has_post_event else 'NO'}")

# 7. Check split report
print("\n[7] Temporal Split:")
with open('out/split_report.json') as f:
    split = json.load(f)
print(f"  Train max month: {split.get('train_max')}")
print(f"  Val range: {split.get('val_min')} to {split.get('val_max')}")
print(f"  Test min month: {split.get('test_min')}")
print(f"  No overlap: {not split.get('overlap', True)}")

# 8. Check runtime
print("\n[8] Runtime Performance:")
with open('out/run_meta.json') as f:
    meta = json.load(f)
print(f"  Elapsed: {meta['elapsed_sec']}s (target: <300s)")
print(f"  Under target: {'YES' if meta['elapsed_sec'] < 300 else 'NO'}")
print(f"  Seed: {meta['seed']}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
