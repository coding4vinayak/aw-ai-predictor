#!/usr/bin/env python3
"""
Sample Data Generator for AI Prediction Platform Testing
Creates comprehensive test data for all ML models with realistic values
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import os

# Create uploads directory if it doesn't exist
os.makedirs('uploads', exist_ok=True)

# Set random seed for reproducible results
np.random.seed(42)
random.seed(42)

def create_lead_scoring_data():
    """Create realistic lead scoring sample data"""
    n_samples = 100
    
    # Company types and industries
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Education', 'Real Estate']
    company_types = ['Startup', 'SMB', 'Enterprise', 'Mid-Market']
    sources = ['Website', 'LinkedIn', 'Cold Call', 'Referral', 'Trade Show', 'Email Campaign']
    
    data = []
    for i in range(n_samples):
        # Generate realistic lead data
        company_size = random.choice(['1-10', '11-50', '51-200', '201-1000', '1000+'])
        industry = random.choice(industries)
        source = random.choice(sources)
        
        # Budget ranges based on company size
        budget_ranges = {
            '1-10': [1000, 25000],
            '11-50': [10000, 100000], 
            '51-200': [50000, 500000],
            '201-1000': [100000, 1000000],
            '1000+': [500000, 5000000]
        }
        
        budget_min, budget_max = budget_ranges[company_size]
        budget = random.randint(budget_min, budget_max)
        
        # Industry-specific scoring
        industry_scores = {
            'Technology': random.uniform(7, 10),
            'Healthcare': random.uniform(6, 9),
            'Finance': random.uniform(5, 8),
            'Retail': random.uniform(4, 7),
            'Manufacturing': random.uniform(3, 6),
            'Education': random.uniform(2, 5),
            'Real Estate': random.uniform(3, 6)
        }
        
        # Engagement score based on source
        engagement_multiplier = {
            'Website': 1.2,
            'LinkedIn': 1.1,
            'Referral': 1.4,
            'Trade Show': 1.3,
            'Email Campaign': 0.9,
            'Cold Call': 0.7
        }
        
        base_engagement = random.uniform(3, 8)
        engagement_score = min(10, base_engagement * engagement_multiplier[source])
        
        # Demographics and behavioral scores
        demographic_score = random.uniform(1, 10)
        behavioral_score = random.uniform(1, 10)
        
        # Source-specific scoring
        source_scores = {
            'Website': random.uniform(6, 10),
            'LinkedIn': random.uniform(5, 8),
            'Referral': random.uniform(8, 10),
            'Trade Show': random.uniform(7, 9),
            'Email Campaign': random.uniform(3, 7),
            'Cold Call': random.uniform(2, 5)
        }
        
        data.append({
            'company_name': f'Company_{i+1}',
            'company_size': company_size,
            'industry': industry,
            'budget': budget,
            'source': source,
            'industry_score': round(industry_scores[industry], 2),
            'engagement_score': round(engagement_score, 2),
            'demographic_score': round(demographic_score, 2),
            'behavioral_score': round(behavioral_score, 2),
            'source_score': round(source_scores[source], 2),
            'lead_quality': random.choice(['Hot', 'Warm', 'Cold'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv('uploads/sample_lead_scoring_data.csv', index=False)
    print(f"✓ Created lead scoring sample data: {len(df)} records")
    return df

def create_churn_prediction_data():
    """Create realistic customer churn data"""
    n_samples = 150
    
    data = []
    for i in range(n_samples):
        # Customer tenure (months)
        tenure = random.randint(1, 72)
        
        # Monthly charges based on tenure and plan type
        if tenure < 12:
            monthly_charges = random.uniform(20, 80)
        elif tenure < 24:
            monthly_charges = random.uniform(40, 120)
        else:
            monthly_charges = random.uniform(60, 200)
        
        # Total charges
        total_charges = monthly_charges * tenure + random.uniform(-500, 1000)
        
        # Contract types
        contract_types = ['Month-to-month', 'One year', 'Two year']
        contract_weights = [0.4, 0.35, 0.25]  # Month-to-month has higher churn
        contract_type = np.random.choice(contract_types, p=contract_weights)
        
        # Payment methods
        payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
        payment_method = random.choice(payment_methods)
        
        # Internet services
        internet_services = ['DSL', 'Fiber optic', 'No internet service']
        internet_service = random.choice(internet_services)
        
        # Support tickets (correlated with churn)
        support_tickets = random.randint(0, 15)
        
        # Additional services
        online_security = random.choice(['Yes', 'No', 'No internet service'])
        online_backup = random.choice(['Yes', 'No', 'No internet service'])
        device_protection = random.choice(['Yes', 'No', 'No internet service'])
        tech_support = random.choice(['Yes', 'No', 'No internet service'])
        streaming_tv = random.choice(['Yes', 'No', 'No internet service'])
        streaming_movies = random.choice(['Yes', 'No', 'No internet service'])
        
        # Demographics
        gender = random.choice(['Male', 'Female'])
        senior_citizen = random.choice([0, 1])
        partner = random.choice(['Yes', 'No'])
        dependents = random.choice(['Yes', 'No'])
        phone_service = random.choice(['Yes', 'No'])
        multiple_lines = random.choice(['Yes', 'No', 'No phone service'])
        paperless_billing = random.choice(['Yes', 'No'])
        
        data.append({
            'customer_id': f'CUST_{i+1:04d}',
            'gender': gender,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure': tenure,
            'phone_service': phone_service,
            'multiple_lines': multiple_lines,
            'internet_service': internet_service,
            'online_security': online_security,
            'online_backup': online_backup,
            'device_protection': device_protection,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
            'contract_type': contract_type,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'monthly_charges': round(monthly_charges, 2),
            'total_charges': round(total_charges, 2),
            'support_tickets': support_tickets
        })
    
    df = pd.DataFrame(data)
    df.to_csv('uploads/sample_churn_prediction_data.csv', index=False)
    print(f"✓ Created churn prediction sample data: {len(df)} records")
    return df

def create_sales_forecast_data():
    """Create realistic sales forecasting data"""
    n_samples = 120
    
    # Generate monthly data for the past 10 years
    start_date = datetime.now() - timedelta(days=365*10)
    
    data = []
    base_sales = 100000
    
    for i in range(n_samples):
        current_date = start_date + timedelta(days=30*i)
        
        # Seasonal patterns
        month = current_date.month
        seasonal_multiplier = {
            1: 0.8, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.15, 8: 1.0, 9: 0.95, 10: 1.05, 11: 1.3, 12: 1.4
        }
        
        # Trend (gradual growth over time)
        trend = 1 + (i * 0.005)
        
        # Random variation
        noise = random.uniform(0.8, 1.2)
        
        # Calculate sales
        historical_sales = base_sales * seasonal_multiplier[month] * trend * noise
        
        # Marketing spend (affects sales)
        marketing_spend = random.uniform(5000, 50000)
        marketing_effect = 1 + (marketing_spend / 100000) * 0.3
        
        # Economic indicators
        economic_index = random.uniform(0.85, 1.15)
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        product_category = random.choice(categories)
        
        # Category-specific adjustments
        category_multipliers = {
            'Electronics': 1.2,
            'Clothing': 1.0,
            'Home & Garden': 0.9,
            'Sports': 1.1,
            'Books': 0.8
        }
        
        final_sales = historical_sales * marketing_effect * economic_index * category_multipliers[product_category]
        
        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'historical_sales': round(historical_sales, 2),
            'marketing_spend': round(marketing_spend, 2),
            'economic_indicators': round(economic_index, 3),
            'product_category': product_category,
            'seasonality': seasonal_multiplier[month],
            'month': month,
            'year': current_date.year,
            'quarter': (month - 1) // 3 + 1,
            'sales_target': round(final_sales, 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('uploads/sample_sales_forecast_data.csv', index=False)
    print(f"✓ Created sales forecast sample data: {len(df)} records")
    return df

def create_nlp_sample_data():
    """Create sample text data for NLP testing"""
    sample_texts = [
        "This product is absolutely amazing! I love everything about it.",
        "The service was terrible and the staff was very rude.",
        "Average product, nothing special but gets the job done.",
        "Outstanding quality and excellent customer support. Highly recommended!",
        "Disappointed with the purchase. Expected much better quality.",
        "Good value for money. Would buy again.",
        "The delivery was fast but the product quality is questionable.",
        "Exceptional experience from start to finish!",
        "Not worth the price. Found better alternatives elsewhere.",
        "Perfect for my needs. Exactly what I was looking for.",
        "Machine learning and artificial intelligence are transforming business operations.",
        "Data analytics provides valuable insights for strategic decision making.",
        "Customer satisfaction scores have improved significantly this quarter.",
        "The new marketing campaign generated substantial lead growth.",
        "Operational efficiency metrics show positive trends across all departments."
    ]
    
    data = []
    for i, text in enumerate(sample_texts):
        data.append({
            'id': i + 1,
            'text': text,
            'source': random.choice(['Review', 'Survey', 'Social Media', 'Email', 'Chat']),
            'category': random.choice(['Product', 'Service', 'Support', 'Marketing', 'General'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv('uploads/sample_nlp_text_data.csv', index=False)
    print(f"✓ Created NLP sample data: {len(df)} records")
    return df

def create_api_test_payloads():
    """Create sample JSON payloads for API testing"""
    
    # Lead scoring payload
    lead_payload = {
        "company_size": "51-200",
        "budget": 150000,
        "industry_score": 8.5,
        "engagement_score": 7.2,
        "demographic_score": 6.8,
        "behavioral_score": 8.1,
        "source_score": 7.5
    }
    
    # Churn prediction payload
    churn_payload = {
        "tenure": 24,
        "monthly_charges": 85.50,
        "total_charges": 2052.00,
        "contract_type": "One year",
        "payment_method": "Credit card",
        "internet_service": "Fiber optic",
        "support_tickets": 2
    }
    
    # Sales forecast payload
    sales_payload = {
        "historical_sales": 125000,
        "seasonality": 1.2,
        "marketing_spend": 25000,
        "economic_indicators": 1.05,
        "product_category": "Electronics"
    }
    
    # NLP payload
    nlp_payload = {
        "text": "This is an excellent product with outstanding customer service. I'm very satisfied with my purchase and would definitely recommend it to others."
    }
    
    payloads = {
        "lead_scoring": lead_payload,
        "churn_prediction": churn_payload,
        "sales_forecast": sales_payload,
        "nlp_analysis": nlp_payload
    }
    
    # Save to JSON file
    with open('uploads/sample_api_payloads.json', 'w') as f:
        json.dump(payloads, f, indent=2)
    
    print("✓ Created API test payloads")
    return payloads

def main():
    """Generate all sample data"""
    print("🔄 Generating comprehensive sample data for AI Prediction Platform...")
    print()
    
    # Create sample datasets
    lead_df = create_lead_scoring_data()
    churn_df = create_churn_prediction_data()
    sales_df = create_sales_forecast_data()
    nlp_df = create_nlp_sample_data()
    
    # Create API payloads
    payloads = create_api_test_payloads()
    
    print()
    print("📊 Sample Data Summary:")
    print(f"  • Lead Scoring: {len(lead_df)} records")
    print(f"  • Churn Prediction: {len(churn_df)} records") 
    print(f"  • Sales Forecast: {len(sales_df)} records")
    print(f"  • NLP Text: {len(nlp_df)} records")
    print()
    print("📁 Files created in uploads/ directory:")
    print("  • sample_lead_scoring_data.csv")
    print("  • sample_churn_prediction_data.csv") 
    print("  • sample_sales_forecast_data.csv")
    print("  • sample_nlp_text_data.csv")
    print("  • sample_api_payloads.json")
    print()
    print("✅ All sample data generated successfully!")
    print()
    print("🚀 Ready for production testing:")
    print("  1. Upload CSV files via /api/upload endpoints")
    print("  2. Test individual predictions using JSON payloads")
    print("  3. Use admin panel to monitor usage and credits")
    print("  4. Test authentication with both API keys and JWT tokens")

if __name__ == "__main__":
    main()