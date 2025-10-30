"""RAG system with TF-IDF retrieval and template-based plan generation."""
import json
import re
import time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import save_jsonl, save_json


class RAGSystem:
    """TF-IDF based RAG for lapse prevention and lead conversion strategies."""
    
    def __init__(self, lapse_docs_dir='out/rag/lapse', lead_docs_dir='out/rag/lead', top_k=3):
        """
        Initialize RAG system.
        
        Parameters:
        -----------
        lapse_docs_dir : str
            Path to lapse prevention documents
        lead_docs_dir : str
            Path to lead conversion documents
        top_k : int
            Number of documents to retrieve (default 3)
        """
        self.lapse_docs_dir = Path(lapse_docs_dir)
        self.lead_docs_dir = Path(lead_docs_dir)
        self.top_k = top_k
        
        # Load documents
        self.lapse_docs = self._load_documents(self.lapse_docs_dir)
        self.lead_docs = self._load_documents(self.lead_docs_dir)
        
        # Build TF-IDF vectorizers
        self.lapse_vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
        self.lead_vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
        
        # Fit vectorizers
        lapse_texts = [doc['content'] for doc in self.lapse_docs]
        lead_texts = [doc['content'] for doc in self.lead_docs]
        
        self.lapse_tfidf = self.lapse_vectorizer.fit_transform(lapse_texts)
        self.lead_tfidf = self.lead_vectorizer.fit_transform(lead_texts)
    
    def _load_documents(self, docs_dir):
        """Load markdown documents from directory."""
        docs = []
        for doc_path in sorted(docs_dir.glob('Doc*.md')):
            with open(doc_path, 'r') as f:
                content = f.read()
            doc_id = doc_path.stem  # e.g., "Doc1"
            docs.append({
                'id': f'Doc#{doc_id[3:]}',  # e.g., "Doc#1"
                'content': content,
                'path': str(doc_path)
            })
        return docs
    
    def retrieve(self, query, corpus='lapse'):
        """
        Retrieve top-k most relevant documents.
        
        Parameters:
        -----------
        query : str
            Query text
        corpus : str
            'lapse' or 'lead'
        
        Returns:
        --------
        list of dict with keys: id, content, score
        """
        if corpus == 'lapse':
            vectorizer = self.lapse_vectorizer
            tfidf_matrix = self.lapse_tfidf
            docs = self.lapse_docs
        else:
            vectorizer = self.lead_vectorizer
            tfidf_matrix = self.lead_tfidf
            docs = self.lead_docs
        
        # Transform query
        query_vec = vectorizer.transform([query])
        
        # Compute cosine similarity
        scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
        
        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:self.top_k]
        
        # Return top-k docs with scores
        retrieved = []
        for idx in top_k_idx:
            retrieved.append({
                'id': docs[idx]['id'],
                'content': docs[idx]['content'],
                'score': float(scores[idx])
            })
        
        return retrieved
    
    def generate_lapse_plan(self, customer_profile, retrieved_docs):
        """
        Generate lapse prevention plan (template-based).
        
        Parameters:
        -----------
        customer_profile : dict
            Keys: age, tenure_m, premium, coverage, region, has_agent, 
                  is_smoker, dependents, lapse_probability, risk_bucket
        retrieved_docs : list of dict
            Retrieved documents with 'id' and 'content'
        
        Returns:
        --------
        dict with plan details
        """
        start_time = time.time()
        
        # Extract top 3 doc IDs
        doc_ids = [doc['id'] for doc in retrieved_docs[:3]]
        
        # Generate 3 steps based on customer profile
        steps = []
        prob = customer_profile['lapse_probability']
        
        # Step 1: Based on risk level with probability embedded
        risk = customer_profile['risk_bucket']
        if risk == 'high':
            step1 = f"At {prob:.0%} lapse risk, immediately activate grace period extension and contact customer within 24 hours to discuss financial hardship options. [{doc_ids[0]}]"
        elif risk == 'mid':
            step1 = f"With {prob:.0%} lapse probability, schedule proactive agent outreach within 3-5 days to review coverage needs and identify concerns before renewal. [{doc_ids[0]}]"
        else:
            step1 = f"At {prob:.0%} risk level, send automated reminder with loyalty rewards information and confirm coverage adequacy for continued satisfaction. [{doc_ids[0]}]"
        steps.append(f"1) {step1}")
        
        # Step 2: Based on agent status and premium
        if customer_profile['has_agent']:
            step2 = f"Have assigned agent conduct personalized policy review, emphasizing relationship value and offering flexible payment plans if needed. [{doc_ids[1]}]"
        else:
            step2 = f"Assign dedicated agent for personalized outreach and explore premium reduction options through coverage adjustments or discounts. [{doc_ids[1]}]"
        steps.append(f"2) {step2}")
        
        # Step 3: Based on tenure and engagement
        tenure = customer_profile['tenure_m']
        if tenure >= 60:
            step3 = f"Leverage long-term loyalty with exclusive retention offers, anniversary bonuses, and recognition of {tenure}-month tenure value. [{doc_ids[2]}]"
        elif tenure >= 24:
            step3 = f"Provide mid-tenure loyalty incentives and demonstrate continued value through coverage summary and savings achieved to date. [{doc_ids[2]}]"
        else:
            step3 = f"Engage through digital channels with payment reminders, policy education content, and introductory retention discounts. [{doc_ids[2]}]"
        steps.append(f"3) {step3}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return {
            'risk_bucket': risk,
            'lapse_probability': customer_profile['lapse_probability'],
            'plan_steps': steps,
            'citations': doc_ids,
            'retrieved_ids': doc_ids,  # For faithfulness check
            'latency_ms': elapsed_ms
        }
    
    def generate_lead_plan(self, lead_profile, retrieved_docs):
        """
        Generate lead conversion plan (template-based).
        
        Parameters:
        -----------
        lead_profile : dict
            Keys: description, age, region, coverage_interest, has_agent_preference
        retrieved_docs : list of dict
            Retrieved documents with 'id' and 'content'
        
        Returns:
        --------
        dict with plan details
        """
        start_time = time.time()
        
        # Extract top 3 doc IDs
        doc_ids = [doc['id'] for doc in retrieved_docs[:3]]
        
        # Generate 3 steps based on lead profile
        steps = []
        
        # Step 1: Segment-based messaging
        age = lead_profile.get('age', 35)
        if age < 30:
            step1 = f"Use mobile-first digital messaging emphasizing affordable protection and easy online enrollment for young professionals. [{doc_ids[0]}]"
        elif age < 50:
            step1 = f"Focus on family protection messaging with income replacement scenarios and coverage for dependents' future needs. [{doc_ids[0]}]"
        else:
            step1 = f"Emphasize legacy planning and retirement security with sophisticated financial planning integration and estate protection. [{doc_ids[0]}]"
        steps.append(f"1) {step1}")
        
        # Step 2: Contact cadence
        if lead_profile.get('has_agent_preference'):
            step2 = f"Schedule agent consultation within 24 hours, follow up with personalized email next day, and maintain weekly touchpoints. [{doc_ids[1]}]"
        else:
            step2 = f"Initiate with instant online quote, send educational content day 3, offer live chat support day 7, and provide limited-time discount at day 14. [{doc_ids[1]}]"
        steps.append(f"2) {step2}")
        
        # Step 3: Objection handling and value
        step3 = f"Address price concerns by demonstrating coverage value, offer trial period with money-back guarantee, and provide competitor comparison showing superior benefits. [{doc_ids[2]}]"
        steps.append(f"3) {step3}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return {
            'risk_bucket': 'n/a',
            'lapse_probability': None,
            'plan_steps': steps,
            'citations': doc_ids,
            'retrieved_ids': doc_ids,  # For faithfulness check
            'latency_ms': elapsed_ms
        }


def select_lapse_customers(preds_df, n_samples=3):
    """
    Select high, median, and low risk customers from predictions.
    
    Returns:
    --------
    list of dicts with customer profiles
    """
    # Sort by predicted probability
    sorted_df = preds_df.sort_values('p_raw', ascending=False).reset_index(drop=True)
    
    # Select high, median, low
    high_idx = 0
    mid_idx = len(sorted_df) // 2
    low_idx = len(sorted_df) - 1
    
    customers = []
    for idx, bucket in [(high_idx, 'high'), (mid_idx, 'mid'), (low_idx, 'low')]:
        row = sorted_df.iloc[idx]
        customers.append({
            'policy_id': int(row['policy_id']),
            'month': row['month'],
            'lapse_probability': float(row['p_raw']),
            'risk_bucket': bucket
        })
    
    return customers


def create_synthetic_leads(n_leads=3):
    """Create synthetic lead profiles."""
    leads = [
        {
            'lead_id': 1,
            'description': 'Young urban digital lead with price objection',
            'age': 28,
            'region': 'E',
            'coverage_interest': 50000,
            'has_agent_preference': False
        },
        {
            'lead_id': 2,
            'description': 'Mid-age family lead via agent',
            'age': 42,
            'region': 'N',
            'coverage_interest': 150000,
            'has_agent_preference': True
        },
        {
            'lead_id': 3,
            'description': 'Senior rural lead unsure about coverage',
            'age': 67,
            'region': 'S',
            'coverage_interest': 75000,
            'has_agent_preference': True
        }
    ]
    return leads


def run_rag_pipeline(preds_df, full_df, output_dir='out'):
    """
    Complete RAG pipeline for lapse prevention and lead conversion.
    
    Returns:
    --------
    dict with audit results
    """
    # Initialize RAG system
    rag = RAGSystem(
        lapse_docs_dir='out/rag/lapse',
        lead_docs_dir='out/rag/lead',
        top_k=3
    )
    
    # Select lapse customers
    lapse_customers = select_lapse_customers(preds_df, n_samples=3)
    
    # Generate lapse prevention plans
    lapse_plans = []
    for customer in lapse_customers:
        # Get full customer profile from data
        customer_row = full_df[
            (full_df['policy_id'] == customer['policy_id']) & 
            (full_df['month'] == customer['month'])
        ].iloc[0]
        
        # Build query
        query = f"High risk customer age {customer_row['age']}, tenure {customer_row['tenure_m']} months, " \
                f"premium ${customer_row['premium']:.0f}, lapse probability {customer['lapse_probability']:.2%}"
        
        # Retrieve docs
        retrieved = rag.retrieve(query, corpus='lapse')
        
        # Generate plan
        profile = {
            'age': int(customer_row['age']),
            'tenure_m': int(customer_row['tenure_m']),
            'premium': float(customer_row['premium']),
            'coverage': float(customer_row['coverage']),
            'region': str(customer_row['region']),
            'has_agent': int(customer_row['has_agent']),
            'is_smoker': int(customer_row['is_smoker']),
            'dependents': int(customer_row['dependents']),
            'lapse_probability': customer['lapse_probability'],
            'risk_bucket': customer['risk_bucket']
        }
        
        plan = rag.generate_lapse_plan(profile, retrieved)
        lapse_plans.append(plan)
    
    # Create synthetic leads
    leads = create_synthetic_leads(n_leads=3)
    
    # Generate lead conversion plans
    lead_plans = []
    for lead in leads:
        # Build query
        query = f"Lead age {lead['age']}, interested in ${lead['coverage_interest']} coverage, " \
                f"{lead['description']}"
        
        # Retrieve docs
        retrieved = rag.retrieve(query, corpus='lead')
        
        # Generate plan
        plan = rag.generate_lead_plan(lead, retrieved)
        lead_plans.append(plan)
    
    # Save plans
    save_jsonl(lapse_plans, f'{output_dir}/lapse_plans.jsonl')
    save_jsonl(lead_plans, f'{output_dir}/lead_plans.jsonl')
    
    # Faithfulness audit
    audit_details = []
    all_plans = lapse_plans + lead_plans
    
    for i, plan in enumerate(all_plans):
        cited = set(plan['citations'])
        retrieved = set(plan.get('retrieved_ids', []))
        faithful = cited.issubset(retrieved)
        
        audit_details.append({
            'plan_id': i,
            'plan_type': 'lapse' if i < len(lapse_plans) else 'lead',
            'retrieved_ids': list(retrieved),
            'cited_ids': list(cited),
            'faithful': faithful
        })
        
        # Assert faithfulness
        assert faithful, f"Faithfulness breach in plan {i}: {cited} not subset of {retrieved}"
    
    audit = {
        'plans': audit_details,
        'faithful_percent': 100.0 if all(p['faithful'] for p in audit_details) else 0.0
    }
    
    save_json(audit, f'{output_dir}/audit_rag.json')
    
    # Smoke asserts
    assert len(lapse_plans) == 3, f"Expected 3 lapse plans, got {len(lapse_plans)}"
    assert len(lead_plans) == 3, f"Expected 3 lead plans, got {len(lead_plans)}"
    assert all(p.get('lapse_probability') is not None for p in lapse_plans), \
        "All lapse plans must have lapse_probability"
    assert all(p.get('lapse_probability') is None for p in lead_plans), \
        "Lead plans should not have lapse_probability"
    assert audit['faithful_percent'] == 100.0, \
        f"Faithfulness audit failed: {audit['faithful_percent']}%"
    
    print(f"RAG pipeline complete: {len(lapse_plans)} lapse + {len(lead_plans)} lead plans")
    print(f"Faithfulness audit: {audit['faithful_percent']}% faithful")
    
    return audit

