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
    rag_top_k = config['rag_top_k']
    
    print("=" * 48)
    print("Decision Assistant pipeline")
    print("=" * 48)
    set_seed(seed)
    print(f"Seed set to {seed} (including PYTHONHASHSEED)")
    
    ensure_dir('out')
    
    with Timer() as timer:
        print("\nGenerating synthetic data…")
        df = generate_synthetic_data(
            n_policies=n_policies,
            n_months_persist=n_months_persist,
            n_months_total=n_months_internal,
            drift_start_month=drift_start,
            seed=seed
        )
        
        df.to_csv('out/data.csv', index=False)
        print(f"  -> {len(df)} rows written to out/data.csv "
              f"({n_policies} policies x {n_months_persist} months)")
        print(f"     Lapse rate: {df['lapse_next_3m'].mean():.3f}")
        
        print("\nSplitting by time and dropping leakage columns…")
        train_df, val_df, test_df, split_report = temporal_split(
            df,
            train_end=config['train_months'][1],
            val_end=config['val_months'][1]
        )
        
        save_json(split_report, 'out/split_report.json')
        print(f"  Train: {len(train_df)} rows (up to {split_report['train_max']})")
        print(f"  Val:   {len(val_df)} rows ({split_report['val_min']} - {split_report['val_max']})")
        print(f"  Test:  {len(test_df)} rows (from {split_report['test_min']})")
        
        train_df, leakage_report = apply_leakage_guard(train_df)
        val_df, _ = apply_leakage_guard(val_df)
        test_df, _ = apply_leakage_guard(test_df)
        
        with open('out/leakage_report.txt', 'w') as f:
            f.write("Leakage Guard Report\n")
            f.write("="*40 + "\n\n")
            f.write(f"Dropped {leakage_report['n_removed']} column(s):\n")
            for col in leakage_report['removed_columns']:
                f.write(f"  - {col}\n")
            f.write("\nThese features are post-event and would cause data leakage.\n")
        
        print(f"  Dropped columns: {leakage_report['removed_columns']}")
        
        print("\nTraining models (random search with early stopping)…")
        print(f"  Trials: {n_trials}")
        
        results = run_modeling_pipeline(
            train_df, val_df, test_df,
            output_dir='out',
            n_trials=n_trials,
            shap_max_samples=shap_max,
            seed=seed
        )
        
        print(
            "  Validation AUC-PR: dummy={:.3f}, logistic={:.3f}, xgb={:.3f}".format(
                results['baseline_results']['dummy_val_aucpr'],
                results['baseline_results']['logreg_val_aucpr'],
                results['best_val_aucpr'],
            )
        )
        print("  Test set metrics:")
        print("    AUC-PR: {:.4f}".format(results['metrics']['auc_pr']))
        print("    ROC-AUC: {:.4f}".format(results['metrics']['roc_auc']))
        print("    Brier: {:.4f}".format(results['metrics']['brier']))
        print("    Precision@1%: {:.4f}".format(results['metrics']['precision_at_1pct']))
        print("    Precision@5%: {:.4f}".format(results['metrics']['precision_at_5pct']))
        print("    Prevalence train/test: {:.3f} / {:.3f}".format(
            results['metrics']['prevalence_train'],
            results['metrics']['prevalence_test'],
        ))
        print("  Saved model.pkl, metrics.json, preds_test.csv, shap_bar.png")
        
        print("\nBuilding RAG plans…")
        audit = run_rag_pipeline(
            results['preds_df'],
            df,  # Use full data for customer profiles
            output_dir='out',
            top_k=rag_top_k
        )
        
        print("\nSaving run metadata…")
        elapsed_sec = timer.elapsed
        
        run_meta = {
            "seed": seed,
            "elapsed_sec": round(elapsed_sec, 2)
        }
        
        save_json(run_meta, 'out/run_meta.json')
        
        if elapsed_sec > 300:
            print(f"  WARNING: Runtime {elapsed_sec:.1f}s exceeds 5-minute target (300s)")
        else:
            print(f"  Runtime: {elapsed_sec:.1f}s (under 5-minute target)")
    
    print("\nSummary")
    print("-" * 48)
    print("Primary metric (AUC-PR): {:.4f}".format(results['metrics']['auc_pr']))
    print("Artifacts in out/: data.csv, split_report.json, leakage_report.txt, metrics.json,")
    print("                   preds_test.csv, model.pkl, shap_bar.png, lapse_plans.jsonl,")
    print("                   lead_plans.jsonl, audit_rag.json, run_meta.json, rag corpora")
    print("-" * 48)


if __name__ == "__main__":
    main()
