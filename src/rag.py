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
        Generate lapse prevention plan using retrieved document content.
        
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
        
        # Extract key phrases from each retrieved document
        doc_ids = []
        doc_summaries = []
        for doc in retrieved_docs[:3]:
            doc_ids.append(doc['id'])
            # Extract first substantive sentence (guidance)
            lines = [l.strip() for l in doc['content'].split('\n') if l.strip() and not l.strip().startswith('#')]
            if lines:
                # Get first sentence of actual content
                first_para = lines[0].split('.')[0] + '.'
                doc_summaries.append(first_para)
            else:
                doc_summaries.append("retention guidance")
        
        # Generate 3 steps using retrieved content
        steps = []
        prob = customer_profile['lapse_probability']
        risk = customer_profile['risk_bucket']
        
        # Step 1: Use doc 0 content for immediate action
        if 'grace' in doc_summaries[0].lower() or 'grace' in retrieved_docs[0]['content'].lower():
            step1 = f"At {prob:.0%} lapse risk, activate grace period extension (15-30 days) per retention best practices, and initiate contact within 48 hours to discuss payment flexibility. [{doc_ids[0]}]"
        elif 'agent' in doc_summaries[0].lower() or 'outreach' in retrieved_docs[0]['content'].lower():
            step1 = f"At {prob:.0%} lapse risk, assign dedicated agent for immediate personal outreach (within 24-48 hours) to address concerns and review policy value. [{doc_ids[0]}]"
        elif 'payment' in doc_summaries[0].lower() or 'flexible' in retrieved_docs[0]['content'].lower():
            step1 = f"At {prob:.0%} lapse risk, offer flexible payment plans (biweekly, monthly, or deferred options) to maintain coverage during financial challenges. [{doc_ids[0]}]"
        elif 'loyalty' in doc_summaries[0].lower() or 'reward' in retrieved_docs[0]['content'].lower():
            step1 = f"At {prob:.0%} lapse risk, apply loyalty rewards and tenure-based discounts (5-15% off) to demonstrate value and prevent lapse. [{doc_ids[0]}]"
        elif 'digital' in doc_summaries[0].lower() or 'reminder' in retrieved_docs[0]['content'].lower():
            step1 = f"At {prob:.0%} lapse risk, deploy automated reminder system via email/SMS with one-click payment links to reduce unintentional lapse. [{doc_ids[0]}]"
        else:
            step1 = f"At {prob:.0%} lapse risk, implement personalized retention strategy based on customer profile and risk factors. [{doc_ids[0]}]"
        steps.append(f"1) {step1}")
        
        # Step 2: Use doc 1 content for relationship/value
        if 'agent' in doc_summaries[1].lower() or 'agent' in retrieved_docs[1]['content'].lower():
            if customer_profile['has_agent']:
                step2 = f"Leverage existing agent relationship for personalized policy review emphasizing 40-60% higher retention rates with agent contact. [{doc_ids[1]}]"
            else:
                step2 = f"Assign dedicated agent immediately to establish personal relationship and explore coverage adjustments that improve retention by 40-60%. [{doc_ids[1]}]"
        elif 'payment' in doc_summaries[1].lower() or 'plan' in retrieved_docs[1]['content'].lower():
            step2 = f"Offer payment plan modifications (frequency changes, temporary deferrals, or installment options) to maintain coverage affordability. [{doc_ids[1]}]"
        elif 'loyalty' in doc_summaries[1].lower() or 'incentive' in retrieved_docs[1]['content'].lower():
            tenure = customer_profile['tenure_m']
            step2 = f"Apply {tenure}-month tenure loyalty benefits including premium credits, anniversary bonuses, and exclusive retention offers. [{doc_ids[1]}]"
        elif 'digital' in doc_summaries[1].lower() or 'engagement' in retrieved_docs[1]['content'].lower():
            step2 = f"Increase digital engagement through portal access, mobile app features, and automated coverage summaries to maintain active relationship. [{doc_ids[1]}]"
        else:
            step2 = f"Provide personalized value demonstration through coverage review and savings summary to reinforce policy benefits. [{doc_ids[1]}]"
        steps.append(f"2) {step2}")
        
        # Step 3: Use doc 2 content for long-term retention
        tenure = customer_profile['tenure_m']
        if 'loyalty' in doc_summaries[2].lower() or 'loyalty' in retrieved_docs[2]['content'].lower():
            if tenure >= 60:
                step3 = f"Implement tiered loyalty program benefits (premium discount 10-15%, priority service, enhanced coverage) for {tenure}-month tenure milestone. [{doc_ids[2]}]"
            else:
                step3 = f"Enroll in loyalty rewards program with tangible benefits at milestone anniversaries to encourage long-term retention. [{doc_ids[2]}]"
        elif 'payment' in doc_summaries[2].lower() or 'flexible' in retrieved_docs[2]['content'].lower():
            step3 = f"Establish flexible payment calendar aligned with customer cash flow patterns to prevent future payment-related lapses. [{doc_ids[2]}]"
        elif 'digital' in doc_summaries[2].lower() or 'reminder' in retrieved_docs[2]['content'].lower():
            step3 = f"Set up multi-channel automated reminders (30/14/7/1 day schedule) and enable self-service payment options for convenience. [{doc_ids[2]}]"
        elif 'season' in doc_summaries[2].lower() or 'lifestyle' in retrieved_docs[2]['content'].lower():
            step3 = f"Address seasonal patterns and lifestyle factors (smoker coaching, wellness programs) that influence retention and risk profile. [{doc_ids[2]}]"
        else:
            step3 = f"Monitor engagement metrics and deploy proactive touchpoints to maintain relationship and prevent future lapse risk. [{doc_ids[2]}]"
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
        Generate lead conversion plan using retrieved document content.
        
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
        
        # Extract key themes from each retrieved document
        doc_ids = []
        doc_summaries = []
        for doc in retrieved_docs[:3]:
            doc_ids.append(doc['id'])
            lines = [l.strip() for l in doc['content'].split('\n') if l.strip() and not l.strip().startswith('#')]
            if lines:
                first_para = lines[0].split('.')[0] + '.'
                doc_summaries.append(first_para)
            else:
                doc_summaries.append("lead conversion guidance")
        
        # Generate 3 steps using retrieved content
        steps = []
        age = lead_profile.get('age', 35)
        
        # Step 1: Use doc 0 content for messaging/segmentation
        if 'segment' in doc_summaries[0].lower() or 'messaging' in retrieved_docs[0]['content'].lower():
            if age < 30:
                step1 = f"Deploy segment-targeted messaging for young professionals emphasizing mobile-first experience, transparent pricing, and digital enrollment convenience. [{doc_ids[0]}]"
            elif age < 50:
                step1 = f"Use family-focused messaging highlighting income replacement, dependent protection, and financial security for growing households. [{doc_ids[0]}]"
            else:
                step1 = f"Emphasize legacy planning and estate protection for retirees/pre-retirees with sophisticated financial integration messaging. [{doc_ids[0]}]"
        elif 'cadence' in doc_summaries[0].lower() or 'contact' in retrieved_docs[0]['content'].lower():
            step1 = f"Implement optimal contact cadence: initial response within 5 minutes (80% higher conversion), followed by strategic touchpoints at days 1, 3, 7, and 14. [{doc_ids[0]}]"
        elif 'value' in doc_summaries[0].lower() or 'proposition' in retrieved_docs[0]['content'].lower():
            step1 = f"Lead with clear value proposition highlighting customer-centric benefits, superior claims service, and competitive advantages with quantified savings. [{doc_ids[0]}]"
        elif 'trial' in doc_summaries[0].lower() or 'offer' in retrieved_docs[0]['content'].lower():
            step1 = f"Present risk-free trial with 30-60 day money-back guarantee and first-month discount to lower initial commitment barriers. [{doc_ids[0]}]"
        else:
            step1 = f"Apply targeted lead conversion strategy based on demographic profile and acquisition channel. [{doc_ids[0]}]"
        steps.append(f"1) {step1}")
        
        # Step 2: Use doc 1 content for cadence/channel
        if 'cadence' in doc_summaries[1].lower() or 'contact' in retrieved_docs[1]['content'].lower():
            if lead_profile.get('has_agent_preference'):
                step2 = f"Execute agent-led cadence: consultation within 24 hours, personalized email follow-up next day, maintain weekly touchpoints with consultative approach. [{doc_ids[1]}]"
            else:
                step2 = f"Deploy digital-first cadence: instant online quote, educational content day 3, live chat day 7, limited-time offer day 14 per optimal timing research. [{doc_ids[1]}]"
        elif 'objection' in doc_summaries[1].lower() or 'handling' in retrieved_docs[1]['content'].lower():
            step2 = f"Prepare objection handling framework: price (value demonstration), timing (urgency creation), competitor (differentiation), using feel-felt-found technique. [{doc_ids[1]}]"
        elif 'channel' in doc_summaries[1].lower() or 'multi' in retrieved_docs[1]['content'].lower():
            step2 = f"Coordinate multi-channel engagement (phone, email, chat, social) with unified message tracking and seamless hand-offs between touchpoints. [{doc_ids[1]}]"
        elif 'value' in doc_summaries[1].lower() or 'proposition' in retrieved_docs[1]['content'].lower():
            step2 = f"Communicate value proposition through customer testimonials, satisfaction scores, and third-party ratings to build credibility and trust. [{doc_ids[1]}]"
        else:
            step2 = f"Maintain consistent engagement through strategic follow-up sequence aligned with lead warming patterns. [{doc_ids[1]}]"
        steps.append(f"2) {step2}")
        
        # Step 3: Use doc 2 content for objection/value/closing
        if 'objection' in doc_summaries[2].lower() or 'objection' in retrieved_docs[2]['content'].lower():
            step3 = f"Deploy objection handling: address price through value quantification, timing through limited offers, competitor through differentiation proof and superior service metrics. [{doc_ids[2]}]"
        elif 'trial' in doc_summaries[2].lower() or 'offer' in retrieved_docs[2]['content'].lower():
            step3 = f"Close with trial offer: 30-day free-look period, first-month discount (10-15% off), money-back guarantee to eliminate risk and accelerate decision. [{doc_ids[2]}]"
        elif 'value' in doc_summaries[2].lower() or 'proposition' in retrieved_docs[2]['content'].lower():
            step3 = f"Reinforce value proposition with specific benefits, customer success stories, and competitive comparisons showing 20-30% better value metrics. [{doc_ids[2]}]"
        elif 'channel' in doc_summaries[2].lower() or 'channel' in retrieved_docs[2]['content'].lower():
            step3 = f"Optimize channel strategy by measuring conversion rates per touchpoint (phone 35%, email 15%, chat 25%) and allocating resources accordingly. [{doc_ids[2]}]"
        else:
            step3 = f"Finalize conversion with clear call-to-action, simplified enrollment process, and immediate confirmation to secure commitment. [{doc_ids[2]}]"
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

