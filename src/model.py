"""Model training, evaluation, and explainability."""
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

from src.utils import save_json, ensure_dir


def temporal_split(df, train_end="2023-08", val_end="2023-10"):
    """
    Split data by prediction month (temporal ordering).
    
    Returns:
    --------
    train_df, val_df, test_df : pd.DataFrame
    split_report : dict
    """
    train_df = df[df['month'] <= train_end].copy()
    val_df = df[(df['month'] > train_end) & (df['month'] <= val_end)].copy()
    test_df = df[df['month'] > val_end].copy()
    
    # Assertions
    train_months = set(train_df['month'].unique())
    val_months = set(val_df['month'].unique())
    test_months = set(test_df['month'].unique())
    
    train_max = max(train_months) if train_months else None
    val_min = min(val_months) if val_months else None
    val_max = max(val_months) if val_months else None
    test_min = min(test_months) if test_months else None
    
    # Compute booleans
    train_max_lt_val_min = train_max < val_min if (train_max and val_min) else False
    val_max_lt_test_min = val_max < test_min if (val_max and test_min) else False
    overlap = bool(
        (train_months & val_months) or 
        (val_months & test_months) or 
        (train_months & test_months)
    )
    
    split_report = {
        "train_max": train_max,
        "val_min": val_min,
        "val_max": val_max,
        "test_min": test_min,
        "train_max<val_min": train_max_lt_val_min,
        "val_max<test_min": val_max_lt_test_min,
        "overlap": overlap,
    }
    
    assert train_max_lt_val_min, "Temporal split failed: train_max >= val_min"
    assert val_max_lt_test_min, "Temporal split failed: val_max >= test_min"
    assert not overlap, "Temporal split failed: overlapping months detected"
    
    return train_df, val_df, test_df, split_report


def apply_leakage_guard(df):
    """
    Drop leakage trap columns (post_event_*).
    
    Returns:
    --------
    df_clean : pd.DataFrame
    leakage_report : dict
    """
    leakage_cols = [col for col in df.columns if col.startswith('post_event_')]
    df_clean = df.drop(columns=leakage_cols)
    
    leakage_report = {
        "dropped_columns": leakage_cols,
        "n_dropped": len(leakage_cols),
    }
    
    return df_clean, leakage_report


def build_preprocessor(numeric_features, categorical_features):
    """Build sklearn ColumnTransformer for preprocessing."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
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
    """Edge-safe precision at top k%."""
    N = len(y_true)
    K = max(1, int(np.ceil(N * k_pct)))
    idx = np.argsort(y_score)[::-1][:K]
    return y_true.iloc[idx].mean() if isinstance(y_true, pd.Series) else y_true[idx].mean()


def train_baseline_models(X_train, y_train, X_val, y_val, preprocessor):
    """Train Dummy and LogisticRegression baselines."""
    results = {}
    
    # Fit preprocessor
    preprocessor.fit(X_train, y_train)
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # Dummy classifier
    dummy = DummyClassifier(strategy='prior', random_state=42)
    dummy.fit(X_train_trans, y_train)
    y_val_pred_dummy = dummy.predict_proba(X_val_trans)[:, 1]
    results['dummy_val_aucpr'] = average_precision_score(y_val, y_val_pred_dummy)
    
    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
    logreg.fit(X_train_trans, y_train)
    y_val_pred_lr = logreg.predict_proba(X_val_trans)[:, 1]
    results['logreg_val_aucpr'] = average_precision_score(y_val, y_val_pred_lr)
    
    return results


def manual_randomized_search(X_train, y_train, X_val, y_val, preprocessor, 
                             n_trials=20, seed=42):
    """
    Manual randomized hyperparameter search (avoids Pipeline + eval_set issue).
    
    Returns:
    --------
    best_clf : XGBClassifier
    best_params : dict
    best_score : float
    """
    # Fit preprocessor on train
    preprocessor.fit(X_train, y_train)
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    # Compute class imbalance weight
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    
    # Build parameter grid (discrete, no scipy distributions)
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
        # Sample random params
        params = {k: np.random.choice(v) for k, v in param_grid.items()}
        params.update({
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": seed,
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
        })
        
        clf = XGBClassifier(**params)
        clf.fit(
            X_train_trans, y_train,
            eval_set=[(X_val_trans, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        y_val_pred = clf.predict_proba(X_val_trans)[:, 1]
        score = average_precision_score(y_val, y_val_pred)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_clf = clf
    
    # Refit best on train with early stopping
    final_clf = XGBClassifier(**best_params)
    final_clf.fit(
        X_train_trans, y_train,
        eval_set=[(X_val_trans, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    return final_clf, best_params, best_score


def compute_test_metrics(y_true, y_pred_proba, y_train):
    """Compute all required test metrics."""
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
    """
    Compute SHAP values on transformed features.
    
    Returns:
    --------
    shap_values : np.ndarray
    feature_names : list
    n_sample : int
    """
    # Sample <= 500 test rows
    n_sample = min(max_samples, len(X_test))
    X_test_sample = X_test.sample(n=n_sample, random_state=seed)
    
    # Transform
    X_test_trans = preprocessor.transform(X_test_sample)
    
    # SHAP
    explainer = shap.TreeExplainer(
        clf,
        feature_perturbation="tree_path_dependent"
    )
    shap_values = explainer.shap_values(
        X_test_trans,
        check_additivity=False
    )
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    
    return shap_values, feature_names, n_sample


def plot_shap_bar(shap_values, feature_names, output_path, top_k=20):
    """Plot SHAP feature importance bar chart."""
    # Mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top-k features (limit to available features)
    actual_top_k = min(top_k, len(feature_names))
    top_k_idx = np.argsort(mean_abs_shap)[::-1][:actual_top_k]
    top_k_features = [feature_names[i] for i in top_k_idx]
    top_k_values = mean_abs_shap[top_k_idx]
    
    # Clean feature names for readability
    top_k_features_clean = []
    for name in top_k_features:
        # Remove transformer prefixes
        if '__' in name:
            name = name.split('__', 1)[1]
        top_k_features_clean.append(name)
    
    # Plot
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
    """Save test predictions to CSV."""
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
    """
    Complete modeling pipeline.
    
    Returns:
    --------
    dict with all outputs
    """
    # Define feature columns
    exclude_cols = ['policy_id', 'month', 'lapse_next_3m']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Separate numeric and categorical
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
    
    # Ensure features exist
    numeric_features = [f for f in numeric_features if f in feature_cols]
    categorical_features = [f for f in categorical_features if f in feature_cols]
    
    # Extract X, y
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df['lapse_next_3m']
    X_val = val_df[numeric_features + categorical_features]
    y_val = val_df['lapse_next_3m']
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df['lapse_next_3m']
    
    # Assert no leakage columns
    assert not any(col.startswith('post_event_') for col in X_train.columns), \
        "Leakage columns found in features!"
    
    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # Train baselines
    print("Training baseline models...")
    baseline_results = train_baseline_models(X_train, y_train, X_val, y_val, preprocessor)
    
    # Train XGB with manual randomized search
    print(f"Training XGBoost with {n_trials} random trials...")
    best_clf, best_params, best_val_aucpr = manual_randomized_search(
        X_train, y_train, X_val, y_val, preprocessor, 
        n_trials=n_trials, seed=seed
    )
    
    # Transform test set
    X_test_trans = preprocessor.transform(X_test)
    
    # Predict on test
    y_test_pred = best_clf.predict_proba(X_test_trans)[:, 1]
    
    # Compute test metrics
    print("Computing test metrics...")
    metrics = compute_test_metrics(y_test, y_test_pred, y_train)
    
    # Save metrics
    save_json(metrics, f'{output_dir}/metrics.json')
    
    # Save predictions
    print("Saving predictions...")
    preds_df = save_predictions(test_df, y_test_pred, f'{output_dir}/preds_test.csv')
    
    # Compute SHAP
    print("Computing SHAP explanations...")
    shap_values, feature_names, n_sample = compute_shap_explanation(
        best_clf, X_test, preprocessor, max_samples=shap_max_samples, seed=seed
    )
    
    # Plot SHAP
    plot_shap_bar(shap_values, feature_names, f'{output_dir}/shap_bar.png', top_k=15)
    
    # Save the trained model
    print("Saving trained model...")
    model_artifact = {
        'preprocessor': preprocessor,
        'classifier': best_clf,
        'params': best_params,
    }
    joblib.dump(model_artifact, f'{output_dir}/model.pkl')
    
    return {
        'preprocessor': preprocessor,
        'best_clf': best_clf,
        'best_params': best_params,
        'best_val_aucpr': best_val_aucpr,
        'baseline_results': baseline_results,
        'metrics': metrics,
        'preds_df': preds_df,
        'shap_sample_size': n_sample,
    }
