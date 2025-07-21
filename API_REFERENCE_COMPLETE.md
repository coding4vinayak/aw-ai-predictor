# Complete API Reference

## Authentication Endpoints

### Register New Account
**POST** `/api/auth/register`

Create a new user account with automatic API key generation.

**Request Body:**
```json
{
  "username": "string (required, 3-50 chars)",
  "email": "string (required, valid email)",
  "password": "string (required, min 6 chars)"
}
```

**Response (201):**
```json
{
  "message": "User registered successfully",
  "user_id": 123,
  "api_key": "uuid-generated-api-key"
}
```

### User Login
**POST** `/api/auth/login`

Authenticate user and receive JWT token.

**Request Body:**
```json
{
  "username": "string (required)",
  "password": "string (required)"
}
```

**Response (200):**
```json
{
  "access_token": "jwt-token",
  "user_id": 123,
  "username": "user123",
  "expires_in": 86400
}
```

### Get User Information
**GET** `/api/auth/user`

Get current user details including API key.

**Headers:**
- `Authorization: Bearer {jwt_token}` OR
- `X-API-Key: {api_key}`

**Response (200):**
```json
{
  "user_id": 123,
  "username": "user123",
  "email": "user@example.com",
  "api_key": "uuid-api-key",
  "credits_remaining": 950,
  "plan": "basic",
  "created_at": "2025-01-15T10:00:00Z"
}
```

### Validate API Key
**GET** `/api/auth/validate`

Validate API key or JWT token.

**Response (200):**
```json
{
  "valid": true,
  "user_id": 123,
  "rate_limit": {
    "remaining": 980,
    "reset": 1642784567
  }
}
```

---

## Core Prediction Endpoints

### Lead Scoring Model
**POST** `/api/predict/lead-score`

Predict lead conversion probability using company and engagement data.

**Headers:**
- `X-API-Key: {api_key}` OR `Authorization: Bearer {jwt_token}`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "company_size": "string (required)",      // "1-10", "11-50", "51-200", "201-1000", "1000+"
  "budget": "number (required)",            // Available budget in USD
  "industry_score": "number (required)",    // Industry relevance score (1.0-10.0)
  "engagement_score": "number (optional)",  // Engagement level (1.0-10.0)
  "demographic_score": "number (optional)", // Demographic fit (1.0-10.0)
  "behavioral_score": "number (optional)",  // Behavioral indicators (1.0-10.0)
  "source_score": "number (optional)"       // Lead source quality (1.0-10.0)
}
```

**Response (200):**
```json
{
  "model": "lead_scoring_v1",
  "prediction": {
    "score": 93.42,                         // Lead score (0-100)
    "quality": "Hot",                       // "Cold", "Warm", "Hot"
    "confidence": 0.9342,                   // Model confidence (0-1)
    "prediction": 1,                        // Binary prediction (0/1)
    "model_version": "v1.0",
    "features_used": ["company_size", "budget", "industry_score"]
  },
  "processing_time": 0.234,
  "credits_consumed": 1,
  "_cached_at": 1642784567                  // Present if result was cached
}
```

### Churn Prediction Model
**POST** `/api/predict/churn`

Predict customer churn probability based on usage patterns.

**Request Body:**
```json
{
  "tenure": "number (required)",            // Months as customer
  "monthly_charges": "number (required)",   // Monthly payment amount
  "contract_type": "string (required)",     // "Month-to-month", "One year", "Two year"
  "payment_method": "string (optional)",    // Payment method
  "internet_service": "string (optional)",  // Service type
  "total_charges": "number (optional)",     // Total charges to date
  "phone_service": "boolean (optional)",    // Has phone service
  "multiple_lines": "string (optional)",    // Multiple lines status
  "online_security": "string (optional)",   // Online security service
  "tech_support": "string (optional)"       // Tech support service
}
```

**Response (200):**
```json
{
  "model": "churn_prediction_v1",
  "prediction": {
    "churn_probability": 0.23,              // Churn probability (0-1)
    "risk_level": "Low",                    // "Low", "Medium", "High"
    "confidence": 0.87,
    "prediction": 0,                        // Binary prediction (0/1)
    "key_factors": ["contract_type", "tenure", "monthly_charges"],
    "retention_score": 77.3                 // Retention likelihood (0-100)
  },
  "processing_time": 0.189,
  "credits_consumed": 1
}
```

### Sales Forecasting Model
**POST** `/api/predict/sales-forecast`

Forecast future sales based on historical data and market factors.

**Request Body:**
```json
{
  "historical_sales": "number (required)",  // Historical sales data
  "seasonality": "number (required)",       // Seasonal factor (0.5-2.0)
  "marketing_spend": "number (required)",   // Marketing investment
  "market_trend": "number (optional)",      // Market growth trend (0.5-2.0)
  "competition_index": "number (optional)", // Competitive pressure (0.1-2.0)
  "economic_indicator": "number (optional)", // Economic conditions (0.5-2.0)
  "promotional_activity": "number (optional)", // Promotional impact (0.0-2.0)
  "sales_team_size": "number (optional)"    // Sales team size
}
```

**Response (200):**
```json
{
  "model": "sales_forecast_v1",
  "prediction": {
    "forecasted_sales": 245673.42,          // Forecasted sales amount
    "confidence_interval": {
      "lower": 220434.18,                   // Lower bound (95% CI)
      "upper": 270912.66                    // Upper bound (95% CI)
    },
    "growth_rate": 0.156,                   // Expected growth rate
    "confidence": 0.91,
    "key_drivers": ["marketing_spend", "seasonality", "market_trend"],
    "risk_factors": ["competition_index"]
  },
  "processing_time": 0.267,
  "credits_consumed": 2
}
```

### NLP Sentiment Analysis
**POST** `/api/predict/sentiment`

Analyze text sentiment and emotional indicators.

**Request Body:**
```json
{
  "text": "string (required, max 10000 chars)"  // Text to analyze
}
```

**Response (200):**
```json
{
  "model": "sentiment_analysis_v1",
  "prediction": {
    "sentiment": "positive",                // "positive", "negative", "neutral"
    "confidence": 0.94,
    "score": 0.87,                         // Sentiment score (-1 to 1)
    "emotions": {
      "joy": 0.72,
      "anger": 0.05,
      "fear": 0.08,
      "sadness": 0.03,
      "surprise": 0.12
    },
    "keywords": ["amazing", "perfect", "excellent"]
  },
  "processing_time": 0.098,
  "credits_consumed": 1
}
```

### NLP Keyword Extraction
**POST** `/api/predict/keywords`

Extract keywords and topics from text.

**Request Body:**
```json
{
  "text": "string (required, max 10000 chars)",  // Text to analyze
  "max_keywords": "number (optional, default 10)" // Maximum keywords to extract
}
```

**Response (200):**
```json
{
  "model": "keyword_extraction_v1",
  "prediction": {
    "keywords": [
      {"word": "machine learning", "score": 0.95, "frequency": 3},
      {"word": "algorithms", "score": 0.87, "frequency": 2},
      {"word": "business efficiency", "score": 0.82, "frequency": 1}
    ],
    "topics": ["technology", "business", "automation"],
    "summary": "Text about machine learning improving business processes",
    "readability_score": 7.8
  },
  "processing_time": 0.156,
  "credits_consumed": 1
}
```

---

## Industry-Specific Endpoints

### Healthcare Models

#### Healthcare Risk Assessment
**POST** `/api/industry/healthcare/risk-assessment`

Assess patient health risks based on medical indicators.

**Request Body:**
```json
{
  "age": "number (required)",               // Patient age
  "gender": "string (required)",            // "male", "female", "other"
  "bmi": "number (required)",               // Body Mass Index
  "blood_pressure_systolic": "number (required)",
  "blood_pressure_diastolic": "number (required)",
  "cholesterol": "number (optional)",       // Cholesterol level
  "diabetes": "boolean (optional)",         // Has diabetes
  "smoking_status": "string (optional)",    // "never", "former", "current"
  "family_history": "object (optional)"     // Family medical history
}
```

#### Treatment Recommendation
**POST** `/api/industry/healthcare/treatment-recommendation`

Recommend treatment options based on patient profile.

#### Patient Outcome Prediction
**POST** `/api/industry/healthcare/patient-outcome`

Predict patient outcomes for treatment plans.

### Finance Models

#### Credit Scoring
**POST** `/api/industry/finance/credit-scoring`

Assess creditworthiness for loan applications.

**Request Body:**
```json
{
  "income": "number (required)",            // Annual income
  "credit_history_length": "number (required)", // Credit history in months
  "existing_debt": "number (required)",     // Current debt amount
  "employment_status": "string (required)", // Employment status
  "loan_amount": "number (required)",       // Requested loan amount
  "collateral_value": "number (optional)",  // Collateral value
  "previous_defaults": "number (optional)"  // Number of previous defaults
}
```

#### Fraud Detection
**POST** `/api/industry/finance/fraud-detection`

Detect potentially fraudulent transactions.

#### Investment Recommendation
**POST** `/api/industry/finance/investment-recommendation`

Provide personalized investment recommendations.

### Retail Models

#### Demand Forecasting
**POST** `/api/industry/retail/demand-forecasting`

Forecast product demand for inventory planning.

#### Price Optimization
**POST** `/api/industry/retail/price-optimization`

Optimize pricing strategies for maximum revenue.

#### Customer Segmentation
**POST** `/api/industry/retail/customer-segmentation`

Segment customers for targeted marketing.

### Manufacturing Models

#### Quality Control
**POST** `/api/industry/manufacturing/quality-control`

Predict product quality based on manufacturing parameters.

#### Predictive Maintenance
**POST** `/api/industry/manufacturing/predictive-maintenance`

Predict equipment maintenance needs.

#### Supply Chain Optimization
**POST** `/api/industry/manufacturing/supply-chain-optimization`

Optimize supply chain operations.

### SaaS Models

#### User Engagement Prediction
**POST** `/api/industry/saas/user-engagement`

Predict user engagement levels.

#### Feature Adoption Prediction
**POST** `/api/industry/saas/feature-adoption`

Predict feature adoption rates.

#### Expansion Revenue Prediction
**POST** `/api/industry/saas/expansion-prediction`

Predict account expansion opportunities.

---

## Enterprise Endpoints

### Batch Processing
**POST** `/api/enterprise/batch/predict`

Process multiple predictions in a single request for high-volume operations.

**Request Body:**
```json
{
  "requests": [
    {
      "model": "string (required)",         // Model name
      "input": "object (required)"          // Model input data
    }
  ]
}
```

**Response (200):**
```json
{
  "results": [
    {
      "index": 0,
      "model": "lead_scoring",
      "prediction": { /* model output */ },
      "processing_time": 0.234,
      "status": "success"
    }
  ],
  "batch_stats": {
    "total_requests": 10,
    "successful": 9,
    "failed": 1,
    "success_rate": 90.0,
    "total_processing_time": 2.45,
    "avg_request_time": 0.245,
    "requests_per_second": 4.08
  },
  "credits_consumed": 10,
  "timestamp": "2025-01-20T15:30:00Z"
}
```

### System Health
**GET** `/api/enterprise/health/detailed`

Comprehensive system health check with detailed metrics.

**Response (200):**
```json
{
  "status": "healthy",                      // "healthy", "degraded", "unhealthy"
  "overall_health_score": 92.5,            // Overall health (0-100)
  "timestamp": "2025-01-20T15:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "latency_ms": 2.5,
      "connections": 15,
      "max_connections": 100
    },
    "cache": {
      "hit_rate": 85.2,
      "size": 1247,
      "max_size": 10000,
      "memory_usage_mb": 245
    },
    "models": {
      "loaded_models": 8,
      "avg_health_score": 89.3,
      "models": { /* individual model health */ }
    },
    "api_performance": {
      "avg_response_time_ms": 156,
      "requests_per_minute": 342,
      "error_rate": 0.12
    }
  },
  "health_factors": {
    "database": 1.0,
    "cache_hit_rate": 0.852,
    "model_health": 0.893
  },
  "uptime": "99.9%",
  "version": "2.0.0"
}
```

### Prometheus Metrics
**GET** `/api/enterprise/metrics/prometheus`

Prometheus-compatible metrics endpoint for monitoring integration.

**Response (200):**
```
# HELP model_predictions_total Total predictions for lead_scoring
# TYPE model_predictions_total counter
model_predictions_total{model="lead_scoring"} 1247

# HELP model_confidence_avg Average confidence for lead_scoring
# TYPE model_confidence_avg gauge
model_confidence_avg{model="lead_scoring"} 0.876

# HELP model_response_time_avg Average response time for lead_scoring
# TYPE model_response_time_avg gauge
model_response_time_avg{model="lead_scoring"} 0.156

# HELP model_health_score Health score for lead_scoring
# TYPE model_health_score gauge
model_health_score{model="lead_scoring"} 89.3

# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total 15674

# HELP cache_hit_rate Cache hit rate percentage
# TYPE cache_hit_rate gauge
cache_hit_rate 85.2
```

### Usage Analytics
**GET** `/api/enterprise/analytics/usage`

Detailed usage analytics and trends.

**Query Parameters:**
- `days`: Number of days to analyze (default: 7, max: 365)
- `model`: Filter by specific model name
- `granularity`: Data granularity ("hour", "day", "week")

**Response (200):**
```json
{
  "analytics": {
    "summary": {
      "total_predictions": 15674,
      "unique_models": 8,
      "success_rate": 97.8,
      "avg_confidence": 0.876,
      "avg_processing_time": 0.234
    },
    "by_model": {
      "lead_scoring": {
        "count": 5230,
        "success_rate": 98.5,
        "avg_confidence": 0.891,
        "avg_processing_time": 0.156
      }
    },
    "by_day": {
      "2025-01-20": 2341,
      "2025-01-19": 2180
    },
    "hourly_distribution": [12, 8, 5, 3, 2, 4, 8, 15, 25, 45, 67, 89],
    "trends": {
      "confidence_trend": [0.87, 0.89, 0.88, 0.91],
      "performance_trend": [0.23, 0.21, 0.19, 0.18],
      "volume_trend": [2100, 2300, 2450, 2600]
    }
  },
  "filters": {
    "days": 7,
    "model": null,
    "start_date": "2025-01-14T00:00:00Z",
    "end_date": "2025-01-20T23:59:59Z"
  },
  "generated_at": "2025-01-20T15:30:00Z"
}
```

### Data Export
**GET** `/api/enterprise/export/predictions`

Export prediction data in various formats.

**Query Parameters:**
- `format`: Export format ("csv", "json")
- `days`: Number of days to export (default: 30, max: 365)
- `model`: Filter by specific model name

**Response (200):**
CSV format with headers:
```
timestamp,model_type,confidence,processing_time,status,prediction_summary
2025-01-20T15:30:00Z,lead_scoring,0.934,0.156,success,93.42
2025-01-20T15:29:45Z,churn_prediction,0.887,0.143,success,0.23
```

### Model Status
**GET** `/api/enterprise/status/models`

Real-time model status and performance metrics.

**Response (200):**
```json
{
  "models": {
    "lead_scoring": {
      "predictions_count": 5230,
      "avg_confidence": 0.891,
      "avg_processing_time": 0.156,
      "error_rate": 1.5,
      "health_score": 89.3,
      "status": "healthy",
      "cache_hit_rate": 73.2,
      "deployment_status": "active",
      "version": "v1.0",
      "last_trained": "2025-01-15T10:00:00Z",
      "training_data_size": 10000,
      "features_count": 7,
      "last_updated": "2025-01-20T15:25:00Z"
    }
  },
  "system_summary": {
    "total_models": 8,
    "healthy_models": 7,
    "avg_health_score": 87.6,
    "cache_performance": {
      "hit_rate": 85.2,
      "size": 1247,
      "total_requests": 15674
    },
    "uptime": "99.9%"
  },
  "alerts": [
    {
      "severity": "medium",
      "message": "Model churn_prediction health score is 68.2%",
      "timestamp": "2025-01-20T15:20:00Z"
    }
  ],
  "timestamp": "2025-01-20T15:30:00Z"
}
```

---

## File Upload Endpoints

### CSV Upload for Lead Scoring
**POST** `/api/upload/csv/lead-scoring`

Upload CSV file for batch lead scoring predictions.

**Headers:**
- `X-API-Key: {api_key}`
- `Content-Type: multipart/form-data`

**Form Data:**
- `file`: CSV file (max 16MB)

**CSV Format:**
```csv
company_size,budget,industry_score,engagement_score
51-200,150000,8.5,7.2
11-50,75000,6.8,5.9
```

**Response (200):**
```json
{
  "message": "File processed successfully",
  "results": [
    {
      "row": 1,
      "prediction": {
        "score": 93.42,
        "quality": "Hot",
        "confidence": 0.9342
      }
    }
  ],
  "summary": {
    "total_rows": 100,
    "processed_rows": 98,
    "failed_rows": 2,
    "success_rate": 98.0,
    "processing_time": 5.67
  },
  "data_quality": {
    "missing_values": 2,
    "invalid_formats": 1,
    "outliers": 3,
    "quality_score": 94.5
  },
  "credits_consumed": 100,
  "file_id": "uuid-file-identifier"
}
```

### CSV Upload for Churn Prediction
**POST** `/api/upload/csv/churn-prediction`

Upload CSV file for batch churn predictions.

### CSV Upload for Sales Forecasting
**POST** `/api/upload/csv/sales-forecast`

Upload CSV file for batch sales forecasting.

---

## Model Gallery Endpoints

### List Available Models
**GET** `/api/model-gallery/models`

Get list of all available models with metadata.

**Response (200):**
```json
{
  "models": [
    {
      "id": 1,
      "name": "lead_scoring",
      "display_name": "Lead Scoring Model",
      "version": "v1.0",
      "status": "active",
      "accuracy": 94.2,
      "created_at": "2025-01-15T10:00:00Z",
      "description": "Predicts lead conversion probability",
      "features": ["company_size", "budget", "industry_score"],
      "training_data_size": 10000,
      "last_retrained": "2025-01-15T10:00:00Z"
    }
  ],
  "total": 8,
  "active": 7
}
```

### Get Model Details
**GET** `/api/model-gallery/models/{model_name}`

Get detailed information about a specific model.

### Activate Model Version
**POST** `/api/model-gallery/models/{model_name}/activate`

Activate a specific version of a model.

**Request Body:**
```json
{
  "version": "v2.0"
}
```

### Upload Custom Model
**POST** `/api/model-gallery/upload`

Upload a custom trained model.

**Form Data:**
- `model_file`: Pickled model file
- `metadata`: JSON metadata

---

## Monitoring Endpoints

### Monitoring Dashboard
**GET** `/monitoring`

Access the real-time monitoring dashboard (web interface).

### API Health Check
**GET** `/monitoring/api/health`

Basic API health status.

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-20T15:30:00Z",
  "version": "2.0.0"
}
```

### Model Metrics
**GET** `/monitoring/api/models`

Get model performance metrics.

### System Metrics
**GET** `/monitoring/api/metrics`

Get system performance metrics.

**Query Parameters:**
- `window`: Time window in minutes (default: 60)

### Active Alerts
**GET** `/monitoring/api/alerts`

Get current system alerts.

---

## Admin Endpoints

### User Management
**GET** `/admin/users`

List all users (admin only).

### Credit Management
**POST** `/admin/users/{user_id}/credits`

Allocate credits to user (admin only).

### System Administration
**POST** `/api/enterprise/admin/cache/clear`

Clear all caches (admin only).

**POST** `/api/enterprise/admin/system/restart`

Restart system components (admin only).

---

## Error Responses

### Standard Error Format
```json
{
  "error": "Error type",
  "message": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": { /* additional error details */ },
  "timestamp": "2025-01-20T15:30:00Z",
  "request_id": "uuid-request-id"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Valid API key or JWT token required |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `INVALID_INPUT` | 422 | Request data validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Request rate limit exceeded |
| `INSUFFICIENT_CREDITS` | 402 | Not enough credits to complete request |
| `MODEL_NOT_FOUND` | 404 | Requested model does not exist |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1642784567
Retry-After: 3600
```

### Rate Limit Configuration

| Limit Type | Requests | Window | Scope |
|------------|----------|--------|-------|
| API Key | 1,000 | 1 hour | Per API key |
| IP Address | 100 | 1 hour | Per IP |
| JWT Token | 5,000 | 1 hour | Per user |
| ML Predictions | 200 | 1 hour | Per user |
| File Uploads | 10 | 1 hour | Per user |

---

## Webhooks

### Webhook Configuration
**POST** `/api/webhooks/configure`

Configure webhook endpoints for events.

### Supported Events
- `prediction.completed`
- `model.updated`
- `alert.triggered`
- `user.credit_low`

### Webhook Payload Example
```json
{
  "event": "prediction.completed",
  "timestamp": "2025-01-20T15:30:00Z",
  "data": {
    "model": "lead_scoring",
    "user_id": 123,
    "prediction_id": "uuid",
    "result": { /* prediction result */ }
  },
  "signature": "hmac-sha256-signature"
}
```

---

This API reference covers all available endpoints in the AI Prediction Platform. For implementation examples and detailed guides, refer to the complete documentation.