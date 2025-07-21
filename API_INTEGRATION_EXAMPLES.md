# AI Prediction Platform - API Integration Examples

## 🔌 Complete Integration Guide for Developers

Your platform provides 74+ prediction endpoints across industries. Here are real-world integration examples with working code.

---

## 🚀 **BASIC API USAGE**

### **Authentication**
```bash
# Get your API key from the dashboard
API_KEY="your-api-key-here"
BASE_URL="https://yourplatform.com/api"
```

### **Simple Prediction Call**
```python
import requests
import json

# Lead scoring prediction
response = requests.post(
    f"{BASE_URL}/predict/lead-score",
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
    },
    json={
        'company_size': 3,
        'budget': 75000,
        'industry_score': 8,
        'engagement_score': 7,
        'demographic_score': 6,
        'behavioral_score': 9,
        'source_score': 7
    }
)

result = response.json()
print(f"Lead Score: {result['prediction']['score']}")
print(f"Quality: {result['prediction']['quality']}")
print(f"Confidence: {result['prediction']['confidence']}")
```

### **Response Format**
```json
{
  "prediction": {
    "score": 96.0,
    "quality": "Hot",
    "confidence": 0.96,
    "prediction": 1,
    "model_version": "v1.0"
  },
  "processing_time": 0.05,
  "model": "lead_scoring_v1",
  "credits_used": 1,
  "credits_remaining": 999
}
```

---

## 🏢 **INDUSTRY-SPECIFIC EXAMPLES**

### **Healthcare - Patient Risk Assessment**
```python
# Predict patient readmission risk
def assess_patient_risk(patient_data):
    response = requests.post(
        f"{BASE_URL}/healthcare/risk-assessment",
        headers={'X-API-Key': API_KEY},
        json={
            'age': patient_data['age'],
            'diagnosis_codes': patient_data['diagnosis'],
            'lab_values': patient_data['labs'],
            'medication_count': patient_data['medications'],
            'previous_admissions': patient_data['history']
        }
    )
    
    if response.status_code == 200:
        risk = response.json()
        return {
            'risk_level': risk['prediction']['risk_level'],
            'probability': risk['prediction']['risk_probability'],
            'recommendations': risk['prediction']['care_recommendations']
        }
    return None

# Usage example
patient = {
    'age': 65,
    'diagnosis': ['diabetes', 'hypertension'],
    'labs': {'glucose': 180, 'bp_systolic': 150},
    'medications': 5,
    'history': 2
}

risk_assessment = assess_patient_risk(patient)
print(f"Risk Level: {risk_assessment['risk_level']}")
```

### **Finance - Fraud Detection**
```javascript
// Real-time transaction fraud check
async function checkTransactionFraud(transaction) {
    try {
        const response = await fetch(`${BASE_URL}/finance/fraud-detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify({
                amount: transaction.amount,
                merchant_type: transaction.merchant,
                location: transaction.location,
                time_of_day: transaction.hour,
                customer_history: transaction.customer_data
            })
        });

        const result = await response.json();
        
        if (result.prediction.fraud_probability > 0.8) {
            // Block transaction
            return {
                action: 'BLOCK',
                reason: 'High fraud risk detected',
                confidence: result.prediction.confidence
            };
        } else if (result.prediction.fraud_probability > 0.5) {
            // Require additional verification
            return {
                action: 'VERIFY',
                reason: 'Moderate fraud risk - verify identity',
                confidence: result.prediction.confidence
            };
        }
        
        return { action: 'APPROVE', confidence: result.prediction.confidence };
        
    } catch (error) {
        console.error('Fraud check failed:', error);
        return { action: 'MANUAL_REVIEW', error: error.message };
    }
}

// Usage in payment processing
const transaction = {
    amount: 5000,
    merchant: 'online_retail',
    location: 'US',
    hour: 23,
    customer_data: { /* customer history */ }
};

const fraudCheck = await checkTransactionFraud(transaction);
console.log(`Action: ${fraudCheck.action}`);
```

### **Retail - Price Optimization**
```python
import pandas as pd

def optimize_product_prices(product_catalog):
    """Optimize prices for entire product catalog"""
    optimized_prices = []
    
    for product in product_catalog:
        response = requests.post(
            f"{BASE_URL}/retail/price-optimization",
            headers={'X-API-Key': API_KEY},
            json={
                'current_price': product['price'],
                'cost': product['cost'],
                'demand_history': product['sales_history'],
                'competitor_prices': product['competitor_data'],
                'seasonality': product['season_factor'],
                'inventory_level': product['stock']
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            optimized_prices.append({
                'product_id': product['id'],
                'current_price': product['price'],
                'optimal_price': result['prediction']['optimal_price'],
                'expected_revenue_lift': result['prediction']['revenue_increase'],
                'confidence': result['prediction']['confidence']
            })
    
    return pd.DataFrame(optimized_prices)

# Usage example
products = [
    {
        'id': 'PROD001',
        'price': 29.99,
        'cost': 15.00,
        'sales_history': [45, 52, 38, 61, 44],
        'competitor_data': [27.99, 31.99, 28.99],
        'season_factor': 1.2,
        'stock': 150
    }
]

pricing_recommendations = optimize_product_prices(products)
print(pricing_recommendations)
```

---

## 🔄 **BATCH PROCESSING**

### **Bulk Customer Analysis**
```python
def process_customer_batch(customers_csv_path, analysis_type='churn'):
    """Process thousands of customers at once"""
    
    # Upload CSV file
    with open(customers_csv_path, 'rb') as file:
        response = requests.post(
            f"{BASE_URL}/batch/{analysis_type}",
            headers={'X-API-Key': API_KEY},
            files={'file': file},
            data={'return_detailed': 'true'}
        )
    
    if response.status_code == 200:
        job = response.json()
        job_id = job['job_id']
        
        # Poll for completion
        while True:
            status_response = requests.get(
                f"{BASE_URL}/batch/status/{job_id}",
                headers={'X-API-Key': API_KEY}
            )
            
            status = status_response.json()
            
            if status['status'] == 'completed':
                # Download results
                results_response = requests.get(
                    f"{BASE_URL}/batch/download/{job_id}",
                    headers={'X-API-Key': API_KEY}
                )
                
                return results_response.json()
            elif status['status'] == 'failed':
                raise Exception(f"Batch job failed: {status['error']}")
            
            time.sleep(10)  # Wait 10 seconds before checking again
    
    raise Exception(f"Failed to submit batch job: {response.text}")

# Usage
results = process_customer_batch('customers.csv', 'churn')
print(f"Processed {len(results)} customers")

# Results include predictions for each customer
for customer in results:
    if customer['churn_probability'] > 0.7:
        print(f"High-risk customer: {customer['customer_id']}")
```

---

## 🔗 **WEBHOOK INTEGRATION**

### **Real-time Prediction Webhooks**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/prediction-complete', methods=['POST'])
def handle_prediction_webhook():
    """Handle completed predictions from your AI platform"""
    
    data = request.get_json()
    
    prediction_id = data.get('prediction_id')
    customer_id = data.get('customer_id')
    model_type = data.get('model_type')
    result = data.get('prediction')
    
    if model_type == 'churn_prediction':
        if result['churn_probability'] > 0.8:
            # Trigger retention campaign
            trigger_retention_email(customer_id)
            schedule_sales_call(customer_id)
            
    elif model_type == 'lead_scoring':
        if result['score'] > 80:
            # Notify sales team of hot lead
            notify_sales_team(customer_id, result['score'])
            
    elif model_type == 'fraud_detection':
        if result['fraud_probability'] > 0.7:
            # Block transaction immediately
            block_transaction(data['transaction_id'])
            send_security_alert(customer_id)
    
    return jsonify({'status': 'processed'})

def trigger_retention_email(customer_id):
    """Send personalized retention email"""
    # Your email marketing logic here
    pass

def notify_sales_team(customer_id, score):
    """Alert sales team about hot lead"""
    # Your CRM integration logic here
    pass

if __name__ == '__main__':
    app.run(port=8080)
```

---

## 📱 **MOBILE APP INTEGRATION**

### **iOS Swift Example**
```swift
import Foundation

class PredictionAPI {
    private let apiKey = "your-api-key"
    private let baseURL = "https://yourplatform.com/api"
    
    func predictChurn(customerData: [String: Any], completion: @escaping (Result<ChurnPrediction, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/predict/churn") else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue(apiKey, forHTTPHeaderField: "X-API-Key")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: customerData)
        } catch {
            completion(.failure(error))
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(APIError.noData))
                return
            }
            
            do {
                let result = try JSONDecoder().decode(ChurnResponse.self, from: data)
                completion(.success(result.prediction))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
}

struct ChurnResponse: Codable {
    let prediction: ChurnPrediction
    let processing_time: Double
    let credits_used: Int
}

struct ChurnPrediction: Codable {
    let churn_probability: Double
    let risk_level: String
    let confidence: Double
    let recommendation: String
}
```

### **Android Kotlin Example**
```kotlin
import retrofit2.Call
import retrofit2.http.*

interface PredictionAPI {
    @POST("predict/lead-score")
    @Headers("Content-Type: application/json")
    fun predictLeadScore(
        @Header("X-API-Key") apiKey: String,
        @Body leadData: LeadScoringRequest
    ): Call<LeadScoringResponse>
}

data class LeadScoringRequest(
    val company_size: Int,
    val budget: Double,
    val industry_score: Int,
    val engagement_score: Int
)

data class LeadScoringResponse(
    val prediction: LeadScore,
    val processing_time: Double,
    val credits_remaining: Int
)

data class LeadScore(
    val score: Double,
    val quality: String,
    val confidence: Double
)

// Usage in Activity/Fragment
class MainActivity : AppCompatActivity() {
    private lateinit var api: PredictionAPI
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Retrofit
        val retrofit = Retrofit.Builder()
            .baseUrl("https://yourplatform.com/api/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            
        api = retrofit.create(PredictionAPI::class.java)
        
        // Make prediction
        val leadData = LeadScoringRequest(
            company_size = 3,
            budget = 75000.0,
            industry_score = 8,
            engagement_score = 7
        )
        
        api.predictLeadScore("your-api-key", leadData).enqueue(object : Callback<LeadScoringResponse> {
            override fun onResponse(call: Call<LeadScoringResponse>, response: Response<LeadScoringResponse>) {
                if (response.isSuccessful) {
                    val leadScore = response.body()?.prediction
                    // Update UI with prediction results
                    updateUI(leadScore)
                }
            }
            
            override fun onFailure(call: Call<LeadScoringResponse>, t: Throwable) {
                // Handle error
            }
        })
    }
}
```

---

## 🔧 **ERROR HANDLING & BEST PRACTICES**

### **Robust Error Handling**
```python
import time
from typing import Optional, Dict, Any

class PredictionClient:
    def __init__(self, api_key: str, base_url: str, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
    
    def predict_with_retry(self, endpoint: str, data: Dict[Any, Any]) -> Optional[Dict]:
        """Make prediction with automatic retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/{endpoint}",
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 402:
                    # Insufficient credits
                    raise InsufficientCreditsError(response.json())
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 500:
                    # Server error - retry
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise ServerError("Server error after retries")
                else:
                    raise APIError(f"API error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts")
        
        return None

# Custom exceptions
class PredictionAPIError(Exception):
    pass

class InsufficientCreditsError(PredictionAPIError):
    def __init__(self, error_data):
        self.credits_needed = error_data.get('credits_needed', 0)
        self.credits_remaining = error_data.get('credits_remaining', 0)
        super().__init__(f"Need {self.credits_needed} credits, have {self.credits_remaining}")

class ServerError(PredictionAPIError):
    pass

class APIError(PredictionAPIError):
    pass

# Usage
client = PredictionClient('your-api-key', 'https://yourplatform.com/api')

try:
    result = client.predict_with_retry('predict/churn', {
        'tenure_months': 24,
        'monthly_charges': 85.50
    })
    print(f"Churn probability: {result['prediction']['churn_probability']}")
    
except InsufficientCreditsError as e:
    print(f"Need to purchase more credits: {e}")
except ServerError as e:
    print(f"Server issues: {e}")
except ConnectionError as e:
    print(f"Connection problems: {e}")
```

---

## 📊 **ANALYTICS & MONITORING**

### **Usage Analytics**
```python
def track_prediction_usage():
    """Get usage analytics for your API integration"""
    
    response = requests.get(
        f"{BASE_URL}/analytics/usage",
        headers={'X-API-Key': API_KEY},
        params={
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'group_by': 'model_type'
        }
    )
    
    if response.status_code == 200:
        analytics = response.json()
        
        for model_stats in analytics['usage_by_model']:
            print(f"Model: {model_stats['model_type']}")
            print(f"  Predictions: {model_stats['total_predictions']}")
            print(f"  Success Rate: {model_stats['success_rate']}%")
            print(f"  Avg Response Time: {model_stats['avg_response_time']}ms")
            print(f"  Credits Used: {model_stats['credits_consumed']}")
            print()
    
    return analytics

# Performance monitoring
def monitor_api_performance():
    """Monitor API response times and accuracy"""
    
    start_time = time.time()
    
    # Test prediction
    response = requests.post(
        f"{BASE_URL}/predict/lead-score",
        headers={'X-API-Key': API_KEY},
        json={'company_size': 2, 'budget': 50000}
    )
    
    response_time = time.time() - start_time
    
    metrics = {
        'response_time_ms': response_time * 1000,
        'status_code': response.status_code,
        'success': response.status_code == 200,
        'timestamp': time.time()
    }
    
    # Log to your monitoring system
    log_metrics_to_datadog(metrics)  # or New Relic, etc.
    
    return metrics
```

---

## 🚀 **PRODUCTION DEPLOYMENT CHECKLIST**

### **Security & Authentication**
- [ ] Store API keys in environment variables
- [ ] Use HTTPS only for all requests
- [ ] Implement request signing for sensitive data
- [ ] Rate limit your own API calls
- [ ] Log API usage for auditing

### **Error Handling**
- [ ] Implement retry logic with exponential backoff
- [ ] Handle all HTTP status codes appropriately
- [ ] Set reasonable request timeouts
- [ ] Graceful degradation when API is unavailable
- [ ] Monitor credit usage and top-up automatically

### **Performance**
- [ ] Cache frequently used predictions
- [ ] Use connection pooling for high-volume usage
- [ ] Implement circuit breaker pattern
- [ ] Monitor response times and set alerts
- [ ] Use batch processing for bulk operations

### **Testing**
- [ ] Unit tests for API integration code
- [ ] Integration tests with staging environment
- [ ] Load testing for expected traffic
- [ ] Error scenario testing
- [ ] Validate all prediction model outputs

**Your platform is ready for production integration!**
**Use these examples to build powerful AI-driven applications.**