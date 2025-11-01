"""Model training and evaluation."""
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

from src.utils import save_json, ensure_dir


def temporal_split(df, train_end="2023-08", val_end="2023-10"):
    """Split data by month for train/val/test."""
    train_df = df[df['month'] <= train_end].copy()
    val_df = df[(df['month'] > train_end) & (df['month'] <= val_end)].copy()
    test_df = df[df['month'] > val_end].copy()
    
    train_months = set(train_df['month'].unique())
    val_months = set(val_df['month'].unique())
    test_months = set(test_df['month'].unique())
    
    train_max = max(train_months) if train_months else None
    val_min = min(val_months) if val_months else None
    val_max = max(val_months) if val_months else None
    test_min = min(test_months) if test_months else None
    
    no_overlap = (
        (train_max < val_min if (train_max and val_min) else False) and
        (val_max < test_min if (val_max and test_min) else False) and
        not bool((train_months & val_months) or (train_months & test_months) or (val_months & test_months))
    )
    
    split_report = {
        "train_months": sorted(train_months),
        "val_months": sorted(val_months),
        "test_months": sorted(test_months),
        "train_max": train_max,
        "val_min": val_min,
        "val_max": val_max,
        "test_min": test_min,
        "no_overlap": no_overlap,
    }
    
    assert no_overlap, "Temporal split failed: overlapping months detected"
    
    return train_df, val_df, test_df, split_report


def apply_leakage_guard(df):
    """Remove post_event_* columns (leakage trap)."""
    leakage_cols = [c for c in df.columns if c.startswith('post_event_')]
    df_clean = df.drop(columns=leakage_cols)
    
    leakage_report = {
        "removed_columns": leakage_cols,
        "n_removed": len(leakage_cols),
    }
    
    return df_clean, leakage_report


def build_preprocessor(numeric_features, categorical_features):
    """Build sklearn preprocessor."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        sparse_threshold=0.3
    )
    
    return preprocessor


def precision_at_k(y_true, y_score, k_pct):
    """Precision at top k%."""
    N = len(y_true)
    K = max(1, int(np.ceil(N * k_pct)))
    idx = np.argsort(y_score)[::-1][:K]
    return y_true.iloc[idx].mean() if isinstance(y_true, pd.Series) else y_true[idx].mean()


def train_baseline_models(X_train, y_train, X_val, y_val, preprocessor):
    """Train Dummy and LogReg baselines for comparison."""
    results = {}
    
    preprocessor.fit(X_train, y_train)
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # Dummy
    dummy = DummyClassifier(strategy='prior', random_state=42)
    dummy.fit(X_train_trans, y_train)
    y_val_pred_dummy = dummy.predict_proba(X_val_trans)[:, 1]
    results['dummy_val_aucpr'] = average_precision_score(y_val, y_val_pred_dummy)
    
    # LogReg
    logreg = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    logreg.fit(X_train_trans, y_train)
    y_val_pred_lr = logreg.predict_proba(X_val_trans)[:, 1]
    results['logreg_val_aucpr'] = average_precision_score(y_val, y_val_pred_lr)
    
    return results


def manual_randomized_search(X_train, y_train, X_val, y_val, preprocessor, n_trials=20, seed=42):
    """Manual random search (avoids Pipeline + eval_set issues)."""
    preprocessor.fit(X_train, y_train)
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # Class imbalance weight
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    
    # Bounds taken from a quick notebook sweep; wide enough to find a solid tree without over-tuning
    param_grid = {
        "n_estimators": [300, 400, 600, 800],
        "learning_rate": np.geomspace(0.02, 0.15, 8).tolist(),
        "max_depth": [3, 4, 6, 8, 10],
        "subsample": np.linspace(0.6, 1.0, 5).tolist(),
        "colsample_bytree": np.linspace(0.6, 1.0, 5).tolist(),
        "reg_lambda": np.linspace(0.0, 5.0, 6).tolist(),
    }
    
    best_score = -np.inf
    best_params = None
    best_clf = None
    
    np.random.seed(seed)
    
    for trial in range(n_trials):
        params = {k: np.random.choice(v) for k, v in param_grid.items()}
        params.update({
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": seed,
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 30,
        })
        
        clf = XGBClassifier(**params)
        clf.fit(
            X_train_trans, y_train,
            eval_set=[(X_val_trans, y_val)],
            verbose=False
        )
        
        y_val_pred = clf.predict_proba(X_val_trans)[:, 1]
        score = average_precision_score(y_val, y_val_pred)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_clf = clf
    
    # Refit on train with best params
    final_clf = XGBClassifier(**best_params)
    final_clf.fit(
        X_train_trans, y_train,
        eval_set=[(X_val_trans, y_val)],
        verbose=False
    )
    
    return final_clf, best_params, best_score


def compute_test_metrics(y_true, y_pred_proba, y_train):
    """Compute test metrics."""
    metrics = {
        "auc_pr": average_precision_score(y_true, y_pred_proba),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "brier": brier_score_loss(y_true, y_pred_proba),
        "precision_at_1pct": precision_at_k(y_true, y_pred_proba, 0.01),
        "precision_at_5pct": precision_at_k(y_true, y_pred_proba, 0.05),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_true.mean()),
    }
    return metrics


def compute_shap_explanation(clf, X_test, preprocessor, max_samples=500, seed=42):
    """Compute SHAP values on transformed features."""
    np.random.seed(seed)
    
    if len(X_test) > max_samples:
        sample_idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[sample_idx]
    else:
        X_sample = X_test
    
    X_sample_trans = preprocessor.transform(X_sample)
    
    explainer = shap.TreeExplainer(
        clf,
        feature_perturbation="tree_path_dependent"
    )
    shap_values = explainer.shap_values(X_sample_trans, check_additivity=False)
    
    feature_names = preprocessor.get_feature_names_out()
    
    return shap_values, feature_names, len(X_sample)


def plot_shap_bar(shap_values, feature_names, output_path, top_k=20):
    """Plot SHAP feature importance bar chart."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    actual_top_k = min(top_k, len(feature_names))
    top_k_idx = np.argsort(mean_abs_shap)[-actual_top_k:][::-1]
    top_k_values = mean_abs_shap[top_k_idx]
    top_k_features = [feature_names[i] for i in top_k_idx]
    
    # Clean feature names
    top_k_features_clean = []
    for name in top_k_features:
        if '__' in name:
            name = name.split('__', 1)[1]
        top_k_features_clean.append(name)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(actual_top_k), top_k_values)
    plt.yticks(range(actual_top_k), top_k_features_clean)
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top {actual_top_k} Feature Importances (SHAP)')
    plt.tight_layout()
    
    ensure_dir(output_path.rsplit('/', 1)[0])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_predictions(test_df, y_pred_proba, output_path):
    """Save predictions to CSV."""
    preds_df = pd.DataFrame({
        'policy_id': test_df['policy_id'].values,
        'month': test_df['month'].values,
        'y_true': test_df['lapse_next_3m'].values,
        'p_raw': y_pred_proba,
    })
    
    ensure_dir(output_path.rsplit('/', 1)[0])
    preds_df.to_csv(output_path, index=False)
    
    return preds_df


def run_modeling_pipeline(train_df, val_df, test_df, output_dir='out', 
                         n_trials=20, shap_max_samples=500, seed=42):
    """Complete modeling pipeline."""
    # Define features
    exclude_cols = ['policy_id', 'month', 'lapse_next_3m']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    numeric_features = [
        'age',
        'tenure_m',
        'premium',
        'coverage',
        'premium_change_90d',
        'dependents',
        'claims_12m',
        'payment_failures',
        'engagement',
    ]
    categorical_features = ['region', 'has_agent', 'is_smoker']
    
    numeric_features = [f for f in numeric_features if f in feature_cols]
    categorical_features = [f for f in categorical_features if f in feature_cols]
    
    # Extract X, y
    X_train = train_df[feature_cols]
    y_train = train_df['lapse_next_3m']
    X_val = val_df[feature_cols]
    y_val = val_df['lapse_next_3m']
    X_test = test_df[feature_cols]
    y_test = test_df['lapse_next_3m']
    
    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # Train baselines
    print("Training baseline models...")
    baseline_results = train_baseline_models(X_train, y_train, X_val, y_val, preprocessor)
    
    # Hyperparameter search
    print(f"Running hyperparameter search ({n_trials} trials)...")
    best_clf, best_params, best_val_aucpr = manual_randomized_search(
        X_train, y_train, X_val, y_val, preprocessor, n_trials=n_trials, seed=seed
    )
    
    # Transform test data
    X_test_trans = preprocessor.transform(X_test)
    
    # Predict
    y_test_pred = best_clf.predict_proba(X_test_trans)[:, 1]
    
    # Test metrics
    print("Computing test metrics...")
    metrics = compute_test_metrics(y_test, y_test_pred, y_train)
    save_json(metrics, f'{output_dir}/metrics.json')
    
    # Predictions
    print("Saving predictions...")
    preds_df = save_predictions(test_df, y_test_pred, f'{output_dir}/preds_test.csv')
    
    # SHAP
    print("Computing SHAP explanations...")
    shap_values, feature_names, n_sample = compute_shap_explanation(
        best_clf, X_test, preprocessor, max_samples=shap_max_samples, seed=seed
    )
    
    plot_shap_bar(shap_values, feature_names, f'{output_dir}/shap_bar.png', top_k=15)
    
    # Save model
    print("Saving trained model...")
    model_artifact = {
        'preprocessor': preprocessor,
        'classifier': best_clf
    }
    joblib.dump(model_artifact, f'{output_dir}/model.pkl')
    
    return {
        'baseline_results': baseline_results,
        'best_params': best_params,
        'best_val_aucpr': best_val_aucpr,
        'metrics': metrics,
        'preds_df': preds_df,
        'shap_sample_size': n_sample,
    }
