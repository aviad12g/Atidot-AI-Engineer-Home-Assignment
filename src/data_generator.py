"""Synthetic insurance panel data generator with temporal structure, drift, and leakage trap."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_data(n_policies=2000, n_months_persist=12, n_months_total=15, 
                           drift_start_month="2023-07", seed=42):
    """
    Generate synthetic insurance panel data.
    
    Parameters:
    -----------
    n_policies : int
        Number of unique policies
    n_months_persist : int
        Number of months to persist to CSV (default 12)
    n_months_total : int
        Total months to generate in memory for label computation (default 15)
    drift_start_month : str
        Month when drift begins (YYYY-MM format)
    seed : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame
        Panel data with only first n_months_persist months
    """
    np.random.seed(seed)
    
    # Generate base policy characteristics (time-invariant)
    policy_ids = np.arange(1, n_policies + 1)
    
    # Age: Normal(45, 15), clipped to [18, 85]
    base_age = np.clip(np.random.normal(45, 15, n_policies), 18, 85).astype(int)
    
    # Region: categorical
    regions = np.random.choice(['N', 'S', 'E', 'W', 'C'], n_policies, 
                              p=[0.25, 0.20, 0.25, 0.15, 0.15])
    
    # Has agent: binary
    has_agent = np.random.binomial(1, 0.35, n_policies)
    
    # Is smoker: binary
    is_smoker = np.random.binomial(1, 0.20, n_policies)
    
    # Dependents: Poisson(lambda=1.2), clipped to [0, 4]
    dependents = np.clip(np.random.poisson(1.2, n_policies), 0, 4)
    
    # Generate monthly panel (all months in memory)
    start_date = datetime(2023, 1, 1)
    months = []
    for i in range(n_months_total):
        # Proper month increment (handles different month lengths)
        year = start_date.year + (start_date.month + i - 1) // 12
        month = (start_date.month + i - 1) % 12 + 1
        months.append(f"{year:04d}-{month:02d}")
    
    # Create panel structure
    records = []
    
    for month_idx, month in enumerate(months):
        for policy_idx, policy_id in enumerate(policy_ids):
            # Tenure increases with month
            tenure_m = max(1, int(np.random.exponential(40)) + month_idx * 1)
            tenure_m = min(tenure_m, 240)  # Cap at 20 years
            
            # Age increases slightly with months (aging)
            age = min(85, base_age[policy_idx] + month_idx // 12)
            
            # Premium: log-normal, influenced by age and smoker status
            base_premium = np.exp(np.random.normal(np.log(150), 0.4))
            age_factor = 1.0 + (age - 45) * 0.01
            smoker_factor = 1.5 if is_smoker[policy_idx] else 1.0
            premium = base_premium * age_factor * smoker_factor
            
            # Coverage: correlated with premium
            coverage = premium * np.random.uniform(50, 150)
            
            # Premium change (last 90 days): small random walk
            premium_change_90d = np.random.normal(0, 0.10)
            premium_change_90d = np.clip(premium_change_90d, -0.25, 0.25)
            
            records.append({
                'policy_id': policy_id,
                'month': month,
                'month_idx': month_idx,  # temporary, for label computation
                'age': age,
                'tenure_m': tenure_m,
                'premium': premium,
                'coverage': coverage,
                'premium_change_90d': premium_change_90d,
                'region': regions[policy_idx],
                'has_agent': has_agent[policy_idx],
                'is_smoker': is_smoker[policy_idx],
                'dependents': dependents[policy_idx],
            })
    
    df = pd.DataFrame(records)
    
    # Compute lapse_next_3m (forward-looking target)
    # For each month m, look at whether policy lapses in m+1, m+2, or m+3
    df = df.sort_values(['policy_id', 'month_idx']).reset_index(drop=True)
    
    # Generate lapse probabilities and actual lapses
    lapse_probs = []
    
    for idx, row in df.iterrows():
        # Base logit components
        logit = -2.0  # baseline
        
        # Age effect (U-shaped: very young and very old higher risk)
        age_dev = abs(row['age'] - 45) / 20.0
        logit += 0.3 * age_dev
        
        # Tenure effect (longer tenure = lower risk)
        logit -= 0.015 * min(row['tenure_m'], 100)
        
        # Premium change effect (increases = higher risk)
        logit += 2.0 * row['premium_change_90d']
        
        # Smoker effect
        if row['is_smoker']:
            logit += 0.4
        
        # Has agent effect (protective)
        if row['has_agent']:
            logit -= 0.6
        
        # Region effects
        region_effects = {'N': 0.0, 'S': 0.2, 'E': -0.1, 'W': 0.1, 'C': 0.0}
        logit += region_effects[row['region']]
        
        # Drift effect (after 2023-07)
        if row['month'] >= drift_start_month:
            logit += 0.2  # +20% base risk increase
            if row['is_smoker']:
                logit += 0.1  # additional smoker effect amplification
        
        # Convert to probability
        prob = 1 / (1 + np.exp(-logit))
        lapse_probs.append(prob)
    
    df['lapse_prob'] = lapse_probs
    
    # Now compute lapse_next_3m for each row
    # This requires looking ahead 3 months for the same policy
    lapse_next_3m = []
    
    for policy_id in df['policy_id'].unique():
        policy_df = df[df['policy_id'] == policy_id].sort_values('month_idx')
        
        for idx, row in policy_df.iterrows():
            month_idx = row['month_idx']
            
            # Look at next 3 months
            future_months = policy_df[
                (policy_df['month_idx'] > month_idx) & 
                (policy_df['month_idx'] <= month_idx + 3)
            ]
            
            if len(future_months) < 3:
                # Not enough future data (this happens for last few months)
                # We'll mark these, but they'll be filtered out for months 1-12
                lapse = 0
            else:
                # Sample whether lapse occurs in any of the next 3 months
                # Use the average prob across the 3 months
                avg_prob = future_months['lapse_prob'].mean()
                lapse = np.random.binomial(1, avg_prob)
            
            lapse_next_3m.append(lapse)
    
    df['lapse_next_3m'] = lapse_next_3m
    
    # Add leakage trap: post_event_call_count
    # This would only be known AFTER a lapse occurs
    # Policies that lapse get high call counts
    df['post_event_call_count'] = df['lapse_next_3m'] * np.random.poisson(5, len(df)) + \
                                  (1 - df['lapse_next_3m']) * np.random.poisson(0.5, len(df))
    df['post_event_call_count'] = df['post_event_call_count'].astype(int)
    
    # Keep only first n_months_persist months for output
    df_persist = df[df['month_idx'] < n_months_persist].copy()
    
    # Drop temporary columns
    df_persist = df_persist.drop(columns=['month_idx', 'lapse_prob'])
    
    # Sort by month, policy_id
    df_persist = df_persist.sort_values(['month', 'policy_id']).reset_index(drop=True)
    
    # Ensure correct dtypes
    df_persist['policy_id'] = df_persist['policy_id'].astype(int)
    df_persist['age'] = df_persist['age'].astype(int)
    df_persist['tenure_m'] = df_persist['tenure_m'].astype(int)
    df_persist['region'] = df_persist['region'].astype(str)
    df_persist['has_agent'] = df_persist['has_agent'].astype(int)
    df_persist['is_smoker'] = df_persist['is_smoker'].astype(int)
    df_persist['dependents'] = df_persist['dependents'].astype(int)
    df_persist['lapse_next_3m'] = df_persist['lapse_next_3m'].astype(int)
    df_persist['post_event_call_count'] = df_persist['post_event_call_count'].astype(int)
    
    return df_persist


if __name__ == "__main__":
    # Test generation
    df = generate_synthetic_data(n_policies=100, n_months_persist=12, n_months_total=15, seed=42)
    print(f"Generated {len(df)} records")
    print(f"Unique policies: {df['policy_id'].nunique()}")
    print(f"Months: {df['month'].unique()}")
    print(f"Lapse rate: {df['lapse_next_3m'].mean():.3f}")
    print("\nSample data:")
    print(df.head(10))

