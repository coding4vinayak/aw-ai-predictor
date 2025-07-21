# Complete API Endpoint Reference

## Authentication Endpoints
```
POST /api/auth/token                    # Get JWT token
POST /api/auth/register                 # Register new user
```

## Core Prediction Endpoints
```
POST /api/predict/lead-score           # Lead scoring (2 credits)
POST /api/predict/churn                # Churn prediction (3 credits)
POST /api/predict/sales-forecast       # Sales forecasting (4 credits)
POST /api/predict/nlp                  # NLP & sentiment analysis (2 credits)
```

## Generic Industry Endpoints
All industries support these standard models:
```
POST /api/industry/{industry}/churn           # Industry-specific churn (3 credits)
POST /api/industry/{industry}/lead-score     # Industry-specific leads (2 credits)
POST /api/industry/{industry}/sales-forecast # Industry-specific sales (4 credits)
POST /api/industry/{industry}/sentiment      # Industry-specific sentiment (2 credits)
GET  /api/industry/industries                # List all supported industries
```

## Specialized Healthcare Endpoints
```
POST /api/specialized/healthcare/risk-assessment    # Clinical risk assessment (4 credits)
POST /api/industry/healthcare/churn                # Healthcare churn prediction (3 credits)
POST /api/industry/healthcare/lead-score           # Healthcare lead scoring (2 credits)
POST /api/industry/healthcare/sentiment            # Healthcare sentiment analysis (2 credits)
```

## Specialized Finance Endpoints
```
POST /api/specialized/finance/fraud-detection      # Advanced fraud detection (5 credits)
POST /api/specialized/finance/credit-scoring       # Credit risk assessment (4 credits)
POST /api/industry/finance/churn                   # Financial churn prediction (3 credits)
POST /api/industry/finance/lead-score              # Financial lead scoring (2 credits)
POST /api/industry/finance/sentiment               # Financial sentiment analysis (2 credits)
```

## Specialized Retail Endpoints
```
POST /api/specialized/retail/price-optimization    # Price optimization (4 credits)
POST /api/specialized/retail/demand-forecast       # Demand forecasting (3 credits)
POST /api/industry/retail/churn                    # Retail churn prediction (3 credits)
POST /api/industry/retail/lead-score               # Retail lead scoring (2 credits)
POST /api/industry/retail/sentiment                # Retail sentiment analysis (2 credits)
```

## Specialized SaaS Endpoints
```
POST /api/specialized/saas/usage-prediction        # Usage pattern prediction (3 credits)
POST /api/specialized/saas/upsell-prediction       # Upsell opportunity detection (4 credits)
POST /api/industry/saas/churn                      # SaaS churn prediction (3 credits)
POST /api/industry/saas/lead-score                 # SaaS lead scoring (2 credits)
POST /api/industry/saas/sentiment                  # SaaS sentiment analysis (2 credits)
```

## Specialized Manufacturing Endpoints
```
POST /api/specialized/manufacturing/quality-prediction    # Quality prediction (4 credits)
POST /api/specialized/manufacturing/maintenance           # Predictive maintenance (5 credits)
POST /api/industry/manufacturing/churn                    # Manufacturing churn (3 credits)
POST /api/industry/manufacturing/lead-score              # Manufacturing leads (2 credits)
POST /api/industry/manufacturing/sentiment               # Manufacturing sentiment (2 credits)
```

## Specialized Education Endpoints
```
POST /api/specialized/education/student-retention        # Student retention prediction (4 credits)
POST /api/specialized/education/performance-prediction   # Academic performance prediction (3 credits)
POST /api/industry/education/churn                       # Education churn prediction (3 credits)
POST /api/industry/education/lead-score                  # Education lead scoring (2 credits)
POST /api/industry/education/sentiment                   # Education sentiment analysis (2 credits)
```

## Specialized Insurance Endpoints
```
POST /api/specialized/insurance/risk-assessment          # Insurance risk assessment (4 credits)
POST /api/specialized/insurance/claim-prediction         # Claim probability prediction (4 credits)
POST /api/industry/insurance/churn                       # Insurance churn prediction (3 credits)
POST /api/industry/insurance/lead-score                  # Insurance lead scoring (2 credits)
POST /api/industry/insurance/sentiment                   # Insurance sentiment analysis (2 credits)
```

## Specialized Real Estate Endpoints
```
POST /api/specialized/real-estate/price-prediction       # Property price prediction (4 credits)
POST /api/specialized/real-estate/market-analysis        # Market analysis (3 credits)
POST /api/industry/real-estate/churn                     # Real estate churn (3 credits)
POST /api/industry/real-estate/lead-score                # Real estate leads (2 credits)
POST /api/industry/real-estate/sentiment                 # Real estate sentiment (2 credits)
```

## File Upload Endpoints
```
POST /api/upload/predictions           # Batch predictions via CSV/Excel upload
POST /api/upload/validate              # Validate file format and data quality
```

## CRM Integration Endpoints
```
GET /api/connectors/hubspot/leads      # Retrieve HubSpot leads
GET /api/connectors/zoho/leads         # Retrieve Zoho CRM leads
```

## Admin Endpoints (Admin Access Required)
```
GET  /admin/api/users                  # List all users
POST /admin/api/users/{id}/credits     # Allocate credits to user
GET  /admin/api/analytics              # Platform analytics
GET  /admin/api/usage-stats            # Usage statistics
POST /admin/api/plans                  # Manage subscription plans
GET  /admin/dashboard                  # Admin dashboard (web)
GET  /admin/users                      # User management (web)
```

## System Health & Information
```
GET /health                            # API health check
GET /api_docs                          # API documentation (web)
GET /api_tester                        # Interactive API tester (web)
GET /getting_started                   # Getting started guide (web)
GET /data_guide                        # Data requirements guide (web)
```

## User Dashboard & Web Interface
```
GET  /                                 # Main dashboard (web)
GET  /login                            # User login (web)
GET  /register                         # User registration (web)
POST /logout                           # User logout
GET  /api/user/api-keys                # Get user's API keys
POST /api/user/api-keys                # Generate new API key
POST /api/toggle-key/{id}              # Toggle API key active status
```

## Supported Industries

The platform supports **8 major industries** with specialized models:

1. **Healthcare** - Clinical risk, patient outcomes, medical sentiment
2. **Finance** - Fraud detection, credit scoring, financial analytics
3. **Retail** - Price optimization, demand forecasting, customer analytics
4. **SaaS** - Usage prediction, upsell opportunities, customer success
5. **Manufacturing** - Quality prediction, predictive maintenance, supply chain
6. **Education** - Student retention, performance prediction, learning analytics
7. **Insurance** - Risk assessment, claim prediction, underwriting
8. **Real Estate** - Price prediction, market analysis, investment scoring

## Credit Costs by Endpoint Type

### Basic Models (2-3 credits)
- Lead scoring, sentiment analysis, basic predictions
- Industry-specific churn and demand forecasting

### Advanced Models (4-5 credits)  
- Risk assessments, specialized predictions, fraud detection
- Price optimization, student retention, quality prediction

### Complex Models (6-8 credits)
- Advanced analytics, predictive maintenance, comprehensive forecasting
- Multi-factor risk models, complex market analysis

## Rate Limits by Subscription Tier

- **Free:** 60 requests/minute, 100 credits/month
- **Basic:** 300 requests/minute, 1,000 credits/month
- **Premium:** 1,000 requests/minute, 5,000 credits/month  
- **Enterprise:** 5,000 requests/minute, 20,000 credits/month

## Response Format

All API endpoints return JSON in this structure:
```json
{
  "prediction": { /* model-specific results */ },
  "processing_time": 0.234,
  "model": "model_name_v1",
  "industry": "industry_name",
  "confidence": 0.92,
  "credits_consumed": 4,
  "credits_remaining": 996
}
```

## Common HTTP Status Codes

- **200** - Success
- **400** - Bad Request (validation error)
- **401** - Unauthorized (invalid/missing API key)
- **402** - Payment Required (insufficient credits)
- **429** - Too Many Requests (rate limit exceeded)
- **500** - Internal Server Error (prediction/system failure)

## Total Endpoint Count

- **Core Endpoints:** 4
- **Generic Industry Endpoints:** 32 (4 models × 8 industries)
- **Specialized Endpoints:** 16 (2 models per industry average)
- **File Upload:** 2
- **CRM Integration:** 2
- **Admin Endpoints:** 6
- **System/Web Endpoints:** 10
- **Authentication:** 2

**Total Available Endpoints: 74**

This comprehensive API provides enterprise-grade machine learning capabilities across all major business industries with flexible authentication, credit-based usage, and both web interface and programmatic access.