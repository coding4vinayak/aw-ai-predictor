# AI Prediction Platform API Documentation

## Overview

The AI Prediction Platform provides comprehensive machine learning services across 8+ industries with credit-based API access, enterprise-grade features, and PostgreSQL database integration. This documentation covers all available endpoints, authentication methods, and industry-specific models.

## Base URL
```
https://your-platform-domain.com
```

## Authentication

The platform supports two authentication methods:

### 1. API Key Authentication (Recommended for integrations)
Include your API key in the request headers:
```
X-API-Key: your-api-key-here
```

### 2. JWT Token Authentication
First obtain a token, then include it in requests:
```
Authorization: Bearer your-jwt-token-here
```

**Get Token:**
```bash
POST /api/auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

## Credit System

All prediction endpoints consume credits based on complexity:
- **Basic models**: 2-3 credits
- **Advanced models**: 4-5 credits
- **Complex models**: 6-8 credits

### Subscription Tiers
- **Free**: 100 credits/month
- **Basic**: 1,000 credits/month ($29/month)
- **Premium**: 5,000 credits/month ($99/month)
- **Enterprise**: 20,000 credits/month ($299/month)

## Core Prediction Endpoints

### 1. Lead Scoring
Predict lead quality and conversion probability.

**Endpoint:** `POST /api/predict/lead-score`  
**Credits:** 2

```bash
curl -X POST https://your-domain.com/api/predict/lead-score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "email": "lead@example.com",
    "company_size": "50-100",
    "industry": "technology",
    "budget": 50000,
    "engagement_score": 75
  }'
```

**Response:**
```json
{
  "prediction": {
    "score": 0.85,
    "category": "high_quality",
    "confidence": 0.92
  },
  "processing_time": 0.234,
  "model": "lead_scoring_v1"
}
```

### 2. Churn Prediction
Predict customer churn probability and risk factors.

**Endpoint:** `POST /api/predict/churn`  
**Credits:** 3

```bash
curl -X POST https://your-domain.com/api/predict/churn \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "customer_id": "12345",
    "tenure_months": 24,
    "monthly_charges": 79.99,
    "total_charges": 1919.76,
    "contract_type": "month_to_month",
    "payment_method": "credit_card"
  }'
```

**Response:**
```json
{
  "prediction": {
    "churn_probability": 0.23,
    "risk_level": "low",
    "confidence": 0.88,
    "key_factors": ["payment_method", "contract_type"]
  },
  "processing_time": 0.187,
  "model": "churn_prediction_v1"
}
```

### 3. Sales Forecasting
Predict future sales and revenue trends.

**Endpoint:** `POST /api/predict/sales-forecast`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/predict/sales-forecast \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "historical_data": [
      {"month": "2024-01", "revenue": 45000},
      {"month": "2024-02", "revenue": 52000},
      {"month": "2024-03", "revenue": 48000}
    ],
    "forecast_months": 6,
    "seasonality": true
  }'
```

### 4. NLP Sentiment Analysis
Analyze text sentiment and extract insights.

**Endpoint:** `POST /api/predict/nlp`  
**Credits:** 2

```bash
curl -X POST https://your-domain.com/api/predict/nlp \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "This product is amazing! Great quality and fast shipping.",
    "analysis_type": "sentiment"
  }'
```

## Industry-Specific Endpoints

### Healthcare Industry

#### Risk Assessment
Advanced healthcare risk prediction with clinical recommendations.

**Endpoint:** `POST /api/specialized/healthcare/risk-assessment`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/healthcare/risk-assessment \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "age": 45,
    "bmi": 26.5,
    "smoking_status": "former",
    "family_history": "diabetes",
    "blood_pressure": "high",
    "cholesterol": "normal"
  }'
```

**Response:**
```json
{
  "prediction": {
    "risk_score": 0.98,
    "risk_category": "critical",
    "confidence": 0.98,
    "primary_risk_factors": ["blood_pressure", "family_history"],
    "care_recommendations": [
      "intensive_monitoring",
      "multidisciplinary_care_team",
      "frequent_follow_ups"
    ],
    "monitoring_frequency": "monthly"
  },
  "processing_time": 3.97,
  "model": "healthcare_risk_v1",
  "industry": "healthcare"
}
```

#### Generic Healthcare Models
- **Churn Prediction:** `POST /api/industry/healthcare/churn` (3 credits)
- **Lead Scoring:** `POST /api/industry/healthcare/lead-score` (2 credits)
- **Sentiment Analysis:** `POST /api/industry/healthcare/sentiment` (2 credits)

### Finance Industry

#### Fraud Detection
Advanced financial fraud detection with risk scoring.

**Endpoint:** `POST /api/specialized/finance/fraud-detection`  
**Credits:** 5

```bash
curl -X POST https://your-domain.com/api/specialized/finance/fraud-detection \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "transaction_amount": 1500.00,
    "merchant_category": "online_retail",
    "location": "New York, NY",
    "time_of_day": "02:30",
    "payment_method": "credit_card",
    "customer_age": 35,
    "account_age_days": 1200
  }'
```

#### Credit Scoring
Comprehensive credit risk assessment.

**Endpoint:** `POST /api/specialized/finance/credit-scoring`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/finance/credit-scoring \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "annual_income": 75000,
    "employment_length": 5,
    "loan_amount": 25000,
    "debt_to_income": 0.35,
    "credit_history_length": 8,
    "number_of_accounts": 12
  }'
```

### Retail Industry

#### Price Optimization
Advanced pricing strategy recommendations.

**Endpoint:** `POST /api/specialized/retail/price-optimization`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/retail/price-optimization \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "product_category": "electronics",
    "current_price": 299.99,
    "competitor_prices": [279.99, 319.99, 289.99],
    "inventory_level": 150,
    "seasonality": "high_season",
    "brand_strength": 0.8
  }'
```

#### Demand Forecasting
Predict product demand and inventory needs.

**Endpoint:** `POST /api/specialized/retail/demand-forecast`  
**Credits:** 3

```bash
curl -X POST https://your-domain.com/api/specialized/retail/demand-forecast \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "product_id": "SKU12345",
    "historical_sales": [45, 52, 38, 67, 59],
    "price_changes": [299.99, 289.99, 299.99, 279.99, 299.99],
    "marketing_campaigns": [false, true, false, true, false],
    "forecast_period": 30
  }'
```

### SaaS Industry

#### Usage Prediction
Predict customer usage patterns and resource needs.

**Endpoint:** `POST /api/specialized/saas/usage-prediction`  
**Credits:** 3

```bash
curl -X POST https://your-domain.com/api/specialized/saas/usage-prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "customer_tier": "premium",
    "monthly_active_users": 1250,
    "feature_adoption_rate": 0.75,
    "support_tickets": 3,
    "integration_count": 8,
    "usage_trend": "growing"
  }'
```

#### Upsell Prediction
Identify upselling opportunities and optimal timing.

**Endpoint:** `POST /api/specialized/saas/upsell-prediction`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/saas/upsell-prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "current_plan": "basic",
    "usage_percentage": 85,
    "feature_requests": 2,
    "customer_satisfaction": 8.5,
    "contract_months_remaining": 4,
    "team_size": 15
  }'
```

### Manufacturing Industry

#### Quality Prediction
Predict product quality and defect probability.

**Endpoint:** `POST /api/specialized/manufacturing/quality-prediction`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/manufacturing/quality-prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "machine_temperature": 85.5,
    "pressure_reading": 12.3,
    "vibration_level": 2.1,
    "production_speed": 150,
    "operator_experience": "expert",
    "maintenance_days_ago": 5
  }'
```

#### Predictive Maintenance
Predict equipment maintenance needs and failure risk.

**Endpoint:** `POST /api/specialized/manufacturing/maintenance`  
**Credits:** 5

```bash
curl -X POST https://your-domain.com/api/specialized/manufacturing/maintenance \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "equipment_age_months": 36,
    "operating_hours": 8760,
    "temperature_variance": 5.2,
    "vibration_increase": 15,
    "last_maintenance_days": 90,
    "performance_degradation": 8
  }'
```

### Education Industry

#### Student Retention
Predict student dropout risk and intervention needs.

**Endpoint:** `POST /api/specialized/education/student-retention`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/education/student-retention \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "gpa": 3.2,
    "attendance_rate": 0.85,
    "financial_aid": true,
    "extracurricular_activities": 2,
    "family_education_level": "college",
    "work_hours_per_week": 15
  }'
```

#### Performance Prediction
Predict academic performance and success probability.

**Endpoint:** `POST /api/specialized/education/performance-prediction`  
**Credits:** 3

```bash
curl -X POST https://your-domain.com/api/specialized/education/performance-prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "previous_gpa": 3.5,
    "study_hours_per_week": 20,
    "course_difficulty": "advanced",
    "prerequisite_completion": true,
    "learning_style": "visual",
    "motivation_score": 8
  }'
```

### Insurance Industry

#### Risk Assessment
Comprehensive insurance risk evaluation.

**Endpoint:** `POST /api/specialized/insurance/risk-assessment`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/insurance/risk-assessment \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "age": 35,
    "gender": "male",
    "occupation": "software_engineer",
    "location": "suburban",
    "driving_record": "clean",
    "credit_score": 750,
    "coverage_amount": 500000
  }'
```

#### Claim Prediction
Predict claim probability and potential costs.

**Endpoint:** `POST /api/specialized/insurance/claim-prediction`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/insurance/claim-prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "policy_type": "auto",
    "customer_age": 28,
    "vehicle_age": 3,
    "annual_mileage": 15000,
    "previous_claims": 0,
    "safety_features": ["airbags", "abs", "backup_camera"]
  }'
```

### Real Estate Industry

#### Price Prediction
Predict property values and market trends.

**Endpoint:** `POST /api/specialized/real-estate/price-prediction`  
**Credits:** 4

```bash
curl -X POST https://your-domain.com/api/specialized/real-estate/price-prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "property_type": "single_family",
    "square_footage": 2500,
    "bedrooms": 4,
    "bathrooms": 3,
    "age": 15,
    "location": "suburban",
    "school_district_rating": 8,
    "recent_renovations": true
  }'
```

#### Market Analysis
Comprehensive real estate market insights.

**Endpoint:** `POST /api/specialized/real-estate/market-analysis`  
**Credits:** 3

```bash
curl -X POST https://your-domain.com/api/specialized/real-estate/market-analysis \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "zip_code": "10001",
    "property_type": "condo",
    "price_range": [300000, 600000],
    "analysis_period": "6_months",
    "include_forecasts": true
  }'
```

## File Upload Endpoints

### Bulk Predictions
Upload CSV/Excel files for batch processing.

**Endpoint:** `POST /api/upload/predictions`  
**Credits:** Variable based on rows

```bash
curl -X POST https://your-domain.com/api/upload/predictions \
  -H "X-API-Key: your-api-key" \
  -F "file=@predictions.csv" \
  -F "model_type=lead_scoring"
```

**Supported Formats:**
- CSV files (up to 16MB)
- Excel files (.xlsx, .xls)
- JSON files

## Industry Endpoint Patterns

All industries support the following generic endpoints:

### Standard Industry Endpoints
- **Churn Prediction:** `POST /api/industry/{industry}/churn`
- **Lead Scoring:** `POST /api/industry/{industry}/lead-score`  
- **Sales Forecast:** `POST /api/industry/{industry}/sales-forecast`
- **Sentiment Analysis:** `POST /api/industry/{industry}/sentiment`

**Supported Industries:**
- `healthcare`
- `finance`
- `retail`
- `saas`
- `manufacturing`
- `education`
- `insurance`
- `real_estate`

### Example Industry-Specific Request
```bash
curl -X POST https://your-domain.com/api/industry/healthcare/churn \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "customer_data": {
      "patient_id": "P12345",
      "last_visit": "2024-01-15",
      "appointment_frequency": "monthly",
      "treatment_adherence": 0.85
    }
  }'
```

## CRM Integrations

### HubSpot Integration
Retrieve leads and contacts from HubSpot CRM.

**Endpoint:** `GET /api/connectors/hubspot/leads`

```bash
curl -X GET https://your-domain.com/api/connectors/hubspot/leads \
  -H "X-API-Key: your-api-key"
```

### Zoho Integration
Access Zoho CRM data and contacts.

**Endpoint:** `GET /api/connectors/zoho/leads`

```bash
curl -X GET https://your-domain.com/api/connectors/zoho/leads \
  -H "X-API-Key: your-api-key"
```

## Admin Endpoints

### User Management (Admin Only)
- `GET /admin/api/users` - List all users
- `POST /admin/api/users/{user_id}/credits` - Allocate credits
- `GET /admin/api/analytics` - Platform analytics

### Credit Management
- `GET /admin/api/usage-stats` - Usage statistics
- `POST /admin/api/plans` - Manage subscription plans

## Response Format

All endpoints return JSON responses with the following structure:

```json
{
  "prediction": {
    // Model-specific prediction results
  },
  "processing_time": 0.234,
  "model": "model_name_v1",
  "industry": "industry_name",
  "confidence": 0.92,
  "credits_consumed": 4,
  "credits_remaining": 996
}
```

## Error Handling

### Common Error Responses

**401 Unauthorized**
```json
{
  "error": "Invalid API key",
  "code": "INVALID_API_KEY"
}
```

**402 Payment Required**
```json
{
  "error": "Insufficient credits",
  "code": "INSUFFICIENT_CREDITS",
  "credits_required": 4,
  "credits_available": 1
}
```

**400 Bad Request**
```json
{
  "error": "Invalid input data",
  "code": "VALIDATION_ERROR",
  "details": {
    "field": "age",
    "message": "Age must be between 0 and 120"
  }
}
```

**500 Internal Server Error**
```json
{
  "error": "Model prediction failed",
  "code": "PREDICTION_ERROR"
}
```

## Rate Limiting

- **Free tier:** 60 requests per minute
- **Basic tier:** 300 requests per minute  
- **Premium tier:** 1000 requests per minute
- **Enterprise tier:** 5000 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Webhooks

Configure webhooks to receive notifications for:
- Prediction completions
- Credit usage alerts
- Model updates
- System maintenance

**Webhook Configuration:**
```bash
POST /api/webhooks
{
  "url": "https://your-domain.com/webhook",
  "events": ["prediction.completed", "credits.low"],
  "secret": "your-webhook-secret"
}
```

## SDKs and Libraries

### Python SDK
```bash
pip install ai-platform-sdk
```

```python
from ai_platform import AIPlatform

client = AIPlatform(api_key="your-api-key")

# Lead scoring
result = client.predict.lead_score({
    "email": "lead@example.com",
    "company_size": "50-100"
})

# Healthcare risk assessment
risk = client.specialized.healthcare.risk_assessment({
    "age": 45,
    "bmi": 26.5
})
```

### JavaScript SDK
```bash
npm install ai-platform-js
```

```javascript
import AIPlatform from 'ai-platform-js';

const client = new AIPlatform('your-api-key');

// Churn prediction
const churn = await client.predict.churn({
  customer_id: '12345',
  tenure_months: 24
});

// Finance fraud detection
const fraud = await client.specialized.finance.fraudDetection({
  transaction_amount: 1500.00,
  merchant_category: 'online_retail'
});
```

## Best Practices

### 1. Efficient Credit Usage
- Batch similar requests using file uploads
- Use appropriate model complexity for your use case
- Cache results when possible for repeated queries

### 2. Error Handling
- Implement exponential backoff for failed requests
- Handle rate limiting gracefully
- Validate input data before sending requests

### 3. Security
- Store API keys securely (environment variables)
- Use HTTPS for all requests
- Implement webhook signature verification

### 4. Performance
- Use concurrent requests for batch processing
- Implement request timeouts
- Monitor response times and adjust accordingly

## Support and Resources

- **API Status:** `GET /health`
- **Documentation:** [https://docs.ai-platform.com](https://docs.ai-platform.com)
- **Support:** support@ai-platform.com
- **Community:** [https://community.ai-platform.com](https://community.ai-platform.com)
- **Status Page:** [https://status.ai-platform.com](https://status.ai-platform.com)

## Changelog

### v2.1.0 (Current)
- Added 16 specialized industry endpoints
- Enhanced PostgreSQL database integration
- Improved credit management system
- Added comprehensive admin panel

### v2.0.0
- Complete industry-specific model coverage
- Credit-based API access system
- Admin user management
- Enhanced data preprocessing

### v1.0.0
- Initial release with core prediction models
- Basic authentication system
- File upload capabilities