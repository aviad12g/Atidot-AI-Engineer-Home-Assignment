"""Generate synthetic insurance panel data."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_data(n_policies=2000, n_months_persist=12, n_months_total=15, 
                           drift_start_month="2023-07", seed=42):
    """Generate synthetic insurance panel data with drift and leakage trap."""
    np.random.seed(seed)
    
    policy_ids = np.arange(1, n_policies + 1)
    
    # Base characteristics (time-invariant)
    base_age = np.clip(np.random.normal(45, 15, n_policies), 18, 85).astype(int)
    regions = np.random.choice(['N', 'S', 'E', 'W', 'C'], n_policies, 
                              p=[0.25, 0.20, 0.25, 0.15, 0.15])
    has_agent = np.random.binomial(1, 0.35, n_policies)
    is_smoker = np.random.binomial(1, 0.20, n_policies)
    dependents = np.clip(np.random.poisson(1.2, n_policies), 0, 4)
    
    # Generate monthly panel
    start_date = datetime(2023, 1, 1)
    months = []
    for i in range(n_months_total):
        year = start_date.year + (start_date.month + i - 1) // 12
        month = (start_date.month + i - 1) % 12 + 1
        months.append(f"{year:04d}-{month:02d}")
    
    records = []
    
    for month_idx, month in enumerate(months):
        for policy_idx, policy_id in enumerate(policy_ids):
            tenure_m = max(1, int(np.random.exponential(40)) + month_idx * 1)
            tenure_m = min(tenure_m, 240)
            
            age = min(85, base_age[policy_idx] + month_idx // 12)
            
            # Premium depends on age + smoker status
            base_premium = np.exp(np.random.normal(np.log(150), 0.4))
            age_factor = 1.0 + (age - 45) * 0.01
            smoker_factor = 1.5 if is_smoker[policy_idx] else 1.0
            premium = base_premium * age_factor * smoker_factor
            
            coverage = premium * np.random.uniform(50, 150)
            
            # Premium change in last 90 days
            if month_idx < 3:
                premium_change_90d = 0.0
            else:
                premium_change_90d = np.random.normal(0.02, 0.08)
                premium_change_90d = np.clip(premium_change_90d, -0.15, 0.30)
            
            # Claims in last 12 months (behavioral signal)
            if tenure_m < 12:
                claims_12m = np.random.poisson(0.1)
            elif tenure_m < 60:
                claims_12m = np.random.poisson(0.4)
            else:
                claims_12m = np.random.poisson(0.2)
            claims_12m = min(claims_12m, 5)
            
            # Payment failures (strong predictor)
            if tenure_m < 6:
                payment_failures = 0
            else:
                base_failure_rate = 0.05
                if is_smoker[policy_idx]:
                    base_failure_rate += 0.03
                if age < 30 or age > 70:
                    base_failure_rate += 0.02
                payment_failures = np.random.binomial(3, base_failure_rate)
            
            # Engagement score
            base_engagement = 0.5
            if has_agent[policy_idx]:
                base_engagement += 0.2
            if tenure_m > 60:
                base_engagement += 0.15
            if dependents[policy_idx] > 2:
                base_engagement += 0.1
            engagement = np.clip(np.random.normal(base_engagement, 0.15), 0.1, 0.95)
            
            records.append({
                'policy_id': policy_id,
                'month': month,
                'month_idx': month_idx,
                'age': age,
                'tenure_m': tenure_m,
                'premium': premium,
                'coverage': coverage,
                'premium_change_90d': premium_change_90d,
                'region': regions[policy_idx],
                'has_agent': has_agent[policy_idx],
                'is_smoker': is_smoker[policy_idx],
                'dependents': dependents[policy_idx],
                'claims_12m': claims_12m,
                'payment_failures': payment_failures,
                'engagement': engagement,
            })
    
    df = pd.DataFrame(records)
    
    # Compute lapse_next_3m target
    df = df.sort_values(['policy_id', 'month_idx']).reset_index(drop=True)
    
    # Group by policy and shift lapse status
    df['lapse_indicator'] = 0
    
    for _, row in df.iterrows():
        policy_id = row['policy_id']
        month_idx = row['month_idx']
        
        # Compute lapse probability
        logit = -2.5  # base risk
        
        # Stronger coefficients for learnable patterns
        logit += -0.8 * row['tenure_m'] / 100.0  # tenure effect (15x stronger)
        logit += 0.8 * row['premium_change_90d'] * 10  # price shock (8x stronger)
        logit += -0.3 if row['has_agent'] else 0.6  # agent protective
        logit += 0.4 if row['is_smoker'] else 0.0
        logit += (row['age'] - 45) / 100.0  # age effect
        logit += -0.15 * row['dependents']
        
        # New features
        if row['claims_12m'] > 2:
            logit += 1.2
        elif row['claims_12m'] == 1:
            logit += 0.3
        
        logit += 3.0 * row['payment_failures']  # very strong
        
        engagement_effect = (1.0 - row['engagement']) * 2.5
        logit += engagement_effect
        
        # Interactions
        if row['premium_change_90d'] > 0.10 and row['tenure_m'] < 24:
            logit += 1.5  # price shock + low tenure
        
        if not row['has_agent'] and row['premium_change_90d'] > 0.08:
            logit += 1.0
        
        if row['is_smoker'] and row['age'] > 60:
            logit += 0.8
        
        if row['has_agent'] and row['tenure_m'] > 120:
            logit -= 1.2  # loyal + agent = very low risk
        
        if row['dependents'] >= 3:
            logit -= 0.5
        
        if row['age'] < 30 and not row['has_agent'] and row['premium_change_90d'] > 0.10:
            logit += 1.8
        
        if row['is_smoker'] and row['premium_change_90d'] > 0.05:
            logit += 1.3
        
        if row['payment_failures'] > 0 and row['engagement'] < 0.4:
            logit += 1.0
        
        if row['claims_12m'] > 1 and not row['has_agent']:
            logit += 0.9
        
        coverage_ratio = row['coverage'] / (row['premium'] * 1000 + 1)
        if coverage_ratio < 0.05:
            logit += 0.7
        
        # Drift starts July (market disruption sim)
        if row['month'] >= drift_start_month:
            logit += 0.8
            if row['is_smoker']:
                logit += 0.4
            if not row['has_agent']:
                logit += 0.3
            if row['region'] in ['S', 'W']:
                logit += 0.5
        
        lapse_prob = 1.0 / (1.0 + np.exp(-logit))
        lapse_prob = np.clip(lapse_prob, 0.01, 0.95)
        
        # Determine if lapse in next 3 months
        will_lapse_next_3m = np.random.binomial(1, lapse_prob)
        
        df.loc[(df['policy_id'] == policy_id) & (df['month_idx'] == month_idx), 'lapse_indicator'] = will_lapse_next_3m
    
    # Compute lapse_next_3m using 3-month lookahead
    df['lapse_next_3m'] = 0
    for policy_id in df['policy_id'].unique():
        policy_df = df[df['policy_id'] == policy_id].sort_values('month_idx')
        
        for idx, row in policy_df.iterrows():
            month_idx = row['month_idx']
            future_rows = policy_df[(policy_df['month_idx'] > month_idx) & 
                                   (policy_df['month_idx'] <= month_idx + 3)]
            
            if len(future_rows) == 3:
                lapsed_in_next_3m = int(future_rows['lapse_indicator'].sum() > 0)
                df.loc[idx, 'lapse_next_3m'] = lapsed_in_next_3m
            else:
                df.loc[idx, 'lapse_next_3m'] = np.nan
    
    df = df.dropna(subset=['lapse_next_3m'])
    df['lapse_next_3m'] = df['lapse_next_3m'].astype(int)
    
    # Add leakage trap
    df['post_event_call_count'] = 0
    for idx, row in df.iterrows():
        if row['lapse_next_3m'] == 1:
            df.loc[idx, 'post_event_call_count'] = np.random.poisson(5)
        else:
            df.loc[idx, 'post_event_call_count'] = np.random.poisson(0.5)
    
    # Keep only first n_months_persist
    df_persist = df[df['month_idx'] < n_months_persist].copy()
    df_persist = df_persist.drop(columns=['month_idx', 'lapse_indicator'])
    
    return df_persist
