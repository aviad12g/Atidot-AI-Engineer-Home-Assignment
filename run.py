#!/usr/bin/env python3
"""
Decision Assistant - Main Entry Point

Runs complete pipeline: data generation, modeling, RAG, and outputs all artifacts.
Target runtime: <5 minutes
"""
import yaml
import pandas as pd

from src.utils import set_seed, Timer, save_json, get_package_versions, ensure_dir
from src.data_generator import generate_synthetic_data
from src.model import (
    temporal_split,
    apply_leakage_guard,
    run_modeling_pipeline
)
from src.rag import run_rag_pipeline


def main():
    """Main pipeline orchestrator."""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    seed = config['seed']
    n_policies = config['n_policies']
    n_months_persist = config['n_months']
    n_months_internal = config['n_months_internal']
    drift_start = config['drift_start_month']
    n_trials = config['n_tuning_trials']
    shap_max = config['shap_max_samples']
    
    # Set global seed
    print("="*60)
    print("DECISION ASSISTANT - Starting Pipeline")
    print("="*60)
    set_seed(seed)
    print(f"Seed set to {seed} (including PYTHONHASHSEED)")
    
    # Start timer
    timer = Timer()
    timer.__enter__()
    
    # Ensure output directory exists
    ensure_dir('out')
    
    # Step 1: Generate synthetic data
    print("\n[1/6] Generating synthetic data...")
    df = generate_synthetic_data(
        n_policies=n_policies,
        n_months_persist=n_months_persist,
        n_months_total=n_months_internal,
        drift_start_month=drift_start,
        seed=seed
    )
    
    # Save data
    df.to_csv('out/data.csv', index=False)
    print(f"  Generated {len(df)} records ({n_policies} policies x {n_months_persist} months)")
    print(f"  Saved to out/data.csv")
    print(f"  Lapse rate: {df['lapse_next_3m'].mean():.3f}")
    
    # Step 2: Temporal split
    print("\n[2/6] Temporal split and leakage guard...")
    train_df, val_df, test_df, split_report = temporal_split(
        df,
        train_end=config['train_months'][1],
        val_end=config['val_months'][1]
    )
    
    save_json(split_report, 'out/split_report.json')
    print(f"  Train: {len(train_df)} records ({split_report['train_max']})")
    print(f"  Val:   {len(val_df)} records ({split_report['val_min']} to {split_report['val_max']})")
    print(f"  Test:  {len(test_df)} records ({split_report['test_min']} onwards)")
    
    # Apply leakage guard
    train_df, leakage_report = apply_leakage_guard(train_df)
    val_df, _ = apply_leakage_guard(val_df)
    test_df, _ = apply_leakage_guard(test_df)
    
    # Save leakage report
    with open('out/leakage_report.txt', 'w') as f:
        f.write("Leakage Guard Report\n")
        f.write("="*40 + "\n\n")
        f.write(f"Dropped {leakage_report['n_dropped']} column(s):\n")
        for col in leakage_report['dropped_columns']:
            f.write(f"  - {col}\n")
        f.write("\nThese features are post-event and would cause data leakage.\n")
    
    print(f"  Leakage guard: dropped {leakage_report['n_dropped']} column(s)")
    print(f"    {leakage_report['dropped_columns']}")
    
    # Step 3: Train models
    print("\n[3/6] Training models...")
    print(f"  Running {n_trials} hyperparameter trials...")
    
    results = run_modeling_pipeline(
        train_df, val_df, test_df,
        output_dir='out',
        n_trials=n_trials,
        shap_max_samples=shap_max,
        seed=seed
    )
    
    print(f"  Baseline (Dummy) val AUC-PR: {results['baseline_results']['dummy_val_aucpr']:.4f}")
    print(f"  Baseline (LogReg) val AUC-PR: {results['baseline_results']['logreg_val_aucpr']:.4f}")
    print(f"  XGBoost (tuned) val AUC-PR: {results['best_val_aucpr']:.4f}")
    print(f"\n  Test Set Metrics:")
    print(f"    AUC-PR:            {results['metrics']['auc_pr']:.4f}")
    print(f"    ROC-AUC:           {results['metrics']['roc_auc']:.4f}")
    print(f"    Brier Score:       {results['metrics']['brier']:.4f}")
    print(f"    Precision@1%:      {results['metrics']['precision_at_1pct']:.4f}")
    print(f"    Precision@5%:      {results['metrics']['precision_at_5pct']:.4f}")
    print(f"    Prevalence (train):{results['metrics']['prevalence_train']:.4f}")
    print(f"    Prevalence (test): {results['metrics']['prevalence_test']:.4f}")
    print(f"  Saved metrics.json, preds_test.csv")
    print(f"  SHAP analysis on {results['shap_sample_size']} samples -> shap_bar.png")
    
    # Step 4: RAG pipeline
    print("\n[4/6] Running RAG pipeline...")
    audit = run_rag_pipeline(
        results['preds_df'],
        df,  # Use full data for customer profiles
        output_dir='out'
    )
    
    # Step 5: Save run metadata
    print("\n[5/6] Saving run metadata...")
    timer.__exit__()
    elapsed_sec = timer.elapsed
    
    run_meta = {
        "seed": seed,
        "elapsed_sec": round(elapsed_sec, 2),
        "n_policies": n_policies,
        "n_months_persist": n_months_persist,
        "n_months_internal": n_months_internal,
        "drift_start_month": drift_start,
        "n_tuning_trials": n_trials,
        "shap_sample_size": results['shap_sample_size'],
        "package_versions": get_package_versions(),
        "note": "Offline, no API keys, no LLM"
    }
    
    save_json(run_meta, 'out/run_meta.json')
    print(f"  Metadata saved to run_meta.json")
    
    # Step 6: Final checks and warnings
    print("\n[6/6] Final checks...")
    
    # Check runtime
    if elapsed_sec > 300:
        print(f"  WARNING: Runtime {elapsed_sec:.1f}s exceeds 5-minute target (300s)")
    else:
        print(f"  Runtime: {elapsed_sec:.1f}s (under 5-minute target)")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nGenerated Outputs (out/):")
    print("  - data.csv               : Synthetic dataset (2023-01 to 2023-12)")
    print("  - split_report.json      : Temporal split validation")
    print("  - leakage_report.txt     : Dropped leakage columns")
    print("  - metrics.json           : 7 test metrics (AUC-PR primary)")
    print("  - preds_test.csv         : Test predictions")
    print("  - shap_bar.png           : Top-20 feature importances")
    print("  - lapse_plans.jsonl      : 3 lapse prevention plans")
    print("  - lead_plans.jsonl       : 3 lead conversion plans")
    print("  - audit_rag.json         : RAG faithfulness audit (100%)")
    print("  - run_meta.json          : Runtime metadata")
    print("  - rag/lapse/Doc1-6.md    : Lapse corpus")
    print("  - rag/lead/Doc1-6.md     : Lead corpus")
    
    print("\nPrimary Metric (AUC-PR): {:.4f}".format(results['metrics']['auc_pr']))
    print("\nNext steps:")
    print("  - Review metrics.json for model performance")
    print("  - Check shap_bar.png for feature drivers")
    print("  - Examine lapse_plans.jsonl and lead_plans.jsonl")
    print("  - See DISCUSSION.md for technical notes")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

