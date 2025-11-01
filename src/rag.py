"""RAG system with TF-IDF retrieval."""
import json
import re
import time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import save_jsonl, save_json


class RAGSystem:
    """TF-IDF based RAG for lapse and lead strategies."""
    
    def __init__(self, lapse_docs_dir='out/rag/lapse', lead_docs_dir='out/rag/lead', top_k=3):
        self.lapse_docs_dir = Path(lapse_docs_dir)
        self.lead_docs_dir = Path(lead_docs_dir)
        self.top_k = top_k
        
        # Load docs
        self.lapse_docs = self._load_documents(self.lapse_docs_dir)
        self.lead_docs = self._load_documents(self.lead_docs_dir)
        
        # Build vectorizers
        self.lapse_vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
        self.lead_vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
        
        lapse_texts = [doc['content'] for doc in self.lapse_docs]
        lead_texts = [doc['content'] for doc in self.lead_docs]
        
        self.lapse_tfidf = self.lapse_vectorizer.fit_transform(lapse_texts)
        self.lead_tfidf = self.lead_vectorizer.fit_transform(lead_texts)
    
    def _load_documents(self, docs_dir):
        """Load markdown files."""
        docs = []
        for doc_path in sorted(docs_dir.glob('Doc*.md')):
            with open(doc_path, 'r') as f:
                content = f.read()
            docs.append({
                'id': doc_path.stem,
                'content': content
            })
        return docs
    
    def retrieve(self, query, corpus='lapse'):
        """Retrieve top-k docs by TF-IDF similarity."""
        # TODO: could experiment with different k values or BM25
        if corpus == 'lapse':
            vectorizer = self.lapse_vectorizer
            tfidf_matrix = self.lapse_tfidf
            docs = self.lapse_docs
        else:
            vectorizer = self.lead_vectorizer
            tfidf_matrix = self.lead_tfidf
            docs = self.lead_docs
        
        query_vec = vectorizer.transform([query])
        scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
        
        top_idx = np.argsort(scores)[-self.top_k:][::-1]
        
        retrieved = []
        for idx in top_idx:
            retrieved.append({
                'id': docs[idx]['id'],
                'content': docs[idx]['content'],
                'score': float(scores[idx])
            })
        
        return retrieved
    
    def generate_lapse_plan(self, profile, retrieved_docs):
        """Generate lapse prevention plan."""
        prob = profile['lapse_probability']
        risk_bucket = profile['risk_bucket']
        
        # Extract doc info
        doc_ids = []
        doc_summaries = []
        for doc in retrieved_docs[:3]:
            doc_ids.append(doc['id'])
            lines = [l.strip() for l in doc['content'].split('\n') if l.strip() and not l.strip().startswith('#')]
            first_sentence = lines[0] if lines else "retention strategy"
            doc_summaries.append(first_sentence.lower()[:100])
        
        # Build risk-specific plan
        if risk_bucket == 'high':
            urgency = "URGENT"
            step1 = f"{urgency}: {prob:.0%} lapse risk detected. Immediate retention protocol. [{doc_ids[0]}]"
            
            if 'grace' in doc_summaries[1] or 'payment' in doc_summaries[1]:
                step2 = f"Offer 30-day grace period + payment plan (3-6 month installments) to ease financial burden. [{doc_ids[1]}]"
            elif 'agent' in doc_summaries[1] or 'outreach' in doc_summaries[1]:
                step2 = f"Escalate to senior agent for personalized outreach within 24h. Phone > email for high-risk. [{doc_ids[1]}]"
            elif 'loyalty' in doc_summaries[1] or 'discount' in doc_summaries[1]:
                step2 = f"Emergency retention offer: 15-20% loyalty discount for 6-month commitment. [{doc_ids[1]}]"
            else:
                step2 = f"Personalized retention offer based on customer profile and payment history. [{doc_ids[1]}]"
            
            if 'season' in doc_summaries[2] or 'smoker' in doc_summaries[2]:
                step3 = f"Smoker-specific coaching: Connect to cessation programs (10-15% premium reduction incentive). [{doc_ids[2]}]"
            else:
                step3 = f"Follow-up cadence: Day 3, Day 7, Day 14 until retention confirmed or lapse finalized. [{doc_ids[2]}]"
        
        elif risk_bucket == 'mid':
            step1 = f"Proactive retention: {prob:.0%} lapse risk. Engage before escalation. [{doc_ids[0]}]"
            
            if 'loyalty' in doc_summaries[1] or 'discount' in doc_summaries[1]:
                step2 = f"Enroll in loyalty program: 5-10% discount, anniversary perks, priority service access. [{doc_ids[1]}]"
            elif 'agent' in doc_summaries[1]:
                step2 = f"Assign dedicated agent for quarterly check-ins and policy optimization reviews. [{doc_ids[1]}]"
            else:
                step2 = f"Offer value-add services: free policy review, coverage optimization, multi-policy bundling. [{doc_ids[1]}]"
            
            step3 = f"Monitor engagement: Track response to offers; escalate if no action within 14 days. [{doc_ids[2]}]"
        
        else:  # low risk
            step1 = f"Preventive care: {prob:.0%} lapse risk (low). Maintain engagement. [{doc_ids[0]}]"
            step2 = f"Standard touchpoint cadence: Quarterly newsletters, annual review invitations, policy updates. [{doc_ids[1]}]"
            step3 = f"Upsell readiness: Track life events (marriage, kids, home purchase) for coverage expansion. [{doc_ids[2]}]"
        
        plan = {
            'policy_id': profile.get('policy_id', None),
            'lapse_probability': prob,
            'risk_bucket': risk_bucket,
            'retrieved_docs': [d['id'] for d in retrieved_docs],
            'plan': [step1, step2, step3]
        }
        
        return plan
    
    def generate_lead_plan(self, lead, retrieved_docs):
        """Generate lead conversion plan."""
        doc_ids = []
        doc_summaries = []
        for doc in retrieved_docs[:3]:
            doc_ids.append(doc['id'])
            lines = [l.strip() for l in doc['content'].split('\n') if l.strip() and not l.strip().startswith('#')]
            first_sentence = lines[0] if lines else "lead strategy"
            doc_summaries.append(first_sentence.lower()[:100])
        
        lead_desc = lead.get('description', '').lower()
        
        # Build lead-specific plan
        if 'young' in lead_desc or lead['age'] < 35:
            step1 = f"Digital-first approach: Instant quote (< 2 min), mobile-optimized signup, no paperwork. [{doc_ids[0]}]"
        elif 'senior' in lead_desc or lead['age'] > 60:
            step1 = f"Trust-building: Agent-led consultation, printed materials, family involvement option. [{doc_ids[0]}]"
        else:
            step1 = f"Hybrid approach: Online quote + optional agent consultation for complex needs. [{doc_ids[0]}]"
        
        if 'price' in lead_desc or 'objection' in lead_desc:
            if 'objection' in doc_summaries[1] or 'handling' in doc_summaries[1]:
                step2 = f"Address price objection: Show cost-benefit analysis, lifetime value, competitor comparison. [{doc_ids[1]}]"
            elif 'discount' in doc_summaries[1] or 'trial' in doc_summaries[1]:
                step2 = f"Intro offer: First month 20% off, 30-day free-look period, no commitment required. [{doc_ids[1]}]"
            else:
                step2 = f"Value emphasis: Demonstrate ROI through real customer scenarios and claims examples. [{doc_ids[1]}]"
        elif 'family' in lead_desc or lead.get('has_agent_preference'):
            step2 = f"Family protection angle: Multi-policy bundling (15-25% savings), dependent coverage, estate planning tie-in. [{doc_ids[1]}]"
        else:
            step2 = f"Segment-specific messaging: Tailor value props to age, income, family status, and regional preferences. [{doc_ids[1]}]"
        
        if 'unsure' in lead_desc or 'coverage' in lead_desc:
            step3 = f"Education-first close: Coverage calculator, needs assessment quiz, expert Q&A session. [{doc_ids[2]}]"
        else:
            if 'trial' in doc_summaries[2] or 'discount' in retrieved_docs[2]['content'].lower():
                step3 = f"Close with trial offer: 30-day free-look, first-month 10-15% off, money-back guarantee. [{doc_ids[2]}]"
            elif 'value' in doc_summaries[2]:
                step3 = f"Reinforce value: Specific benefits, customer stories, 20-30% better value vs competitors. [{doc_ids[2]}]"
            else:
                step3 = f"Multi-channel follow-up: Phone (35% conversion), email (15%), chat (25%). [{doc_ids[2]}]"
        
        plan = {
            'lead_id': lead['lead_id'],
            'description': lead['description'],
            'retrieved_docs': [d['id'] for d in retrieved_docs],
            'plan': [step1, step2, step3]
        }
        
        return plan


def pick_demo_customers(preds_df, n_customers=3):
    """Select high/mid/low risk customers from predictions."""
    sorted_df = preds_df.sort_values('p_raw', ascending=False)
    
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
    """Create lead profiles."""
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


def run_rag_pipeline(preds_df, full_df, output_dir='out', top_k=3):
    """Run RAG for lapse prevention and lead conversion."""
    # Initialize RAG
    rag = RAGSystem(
        lapse_docs_dir=f'{output_dir}/rag/lapse',
        lead_docs_dir=f'{output_dir}/rag/lead',
        top_k=top_k
    )
    
    # Pick demo customers
    demo_customers = pick_demo_customers(preds_df, n_customers=3)
    
    # Generate lapse plans
    lapse_plans = []
    for customer in demo_customers:
        customer_row = full_df[
            (full_df['policy_id'] == customer['policy_id']) & 
            (full_df['month'] == customer['month'])
        ].iloc[0]
        
        # Build query with risk bucket
        risk_label = customer['risk_bucket'].capitalize()
        query = (
            f"{risk_label} risk customer age {customer_row['age']}, tenure {customer_row['tenure_m']} months, "
            f"premium ${customer_row['premium']:.0f}, lapse probability {customer['lapse_probability']:.2%}"
        )
        
        retrieved = rag.retrieve(query, corpus='lapse')
        
        profile = {
            'policy_id': int(customer_row['policy_id']),
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
    
    # Generate lead plans
    leads = create_synthetic_leads(n_leads=3)
    lead_plans = []
    for lead in leads:
        query = f"Lead conversion: {lead['description']}, age {lead['age']}, region {lead['region']}"
        retrieved = rag.retrieve(query, corpus='lead')
        
        plan = rag.generate_lead_plan(lead, retrieved)
        lead_plans.append(plan)
    
    # Save plans
    # Format with rendered text for audit
    lapse_plans_output = []
    for p in lapse_plans:
        rendered = '\n'.join(p['plan'])
        lapse_plans_output.append({
            'policy_id': p['policy_id'],
            'lapse_probability': p['lapse_probability'],
            'risk_bucket': p['risk_bucket'],
            'retrieved_docs': p['retrieved_docs'],
            'plan': p['plan'],
            'rendered_text': rendered
        })
    
    lead_plans_output = []
    for p in lead_plans:
        rendered = '\n'.join(p['plan'])
        lead_plans_output.append({
            'lead_id': p['lead_id'],
            'description': p['description'],
            'retrieved_docs': p['retrieved_docs'],
            'plan': p['plan'],
            'rendered_text': rendered
        })
    
    save_jsonl(lapse_plans_output, f'{output_dir}/lapse_plans.jsonl')
    save_jsonl(lead_plans_output, f'{output_dir}/lead_plans.jsonl')
    
    # Faithfulness audit
    audit_results = []
    
    for plan in lapse_plans_output + lead_plans_output:
        rendered_text = plan['rendered_text']
        retrieved_ids = set(plan['retrieved_docs'])
        
        cited_ids = set(re.findall(r'\[(Doc\d+)\]', rendered_text))
        
        faithful = cited_ids.issubset(retrieved_ids)
        
        plan_id = plan.get('policy_id', plan.get('lead_id'))
        audit_results.append({
            'plan_id': plan_id,
            'retrieved_ids': sorted(retrieved_ids),
            'cited_ids': sorted(cited_ids),
            'faithful': faithful
        })
    
    faithful_count = sum(1 for r in audit_results if r['faithful'])
    faithful_pct = 100.0 * faithful_count / len(audit_results) if audit_results else 0.0
    
    audit = {
        'plans': audit_results,
        'summary': {
            'total_plans': len(audit_results),
            'faithful_plans': faithful_count,
            'faithful_pct': faithful_pct
        }
    }
    
    save_json(audit, f'{output_dir}/audit_rag.json')
    
    return audit
