#!/usr/bin/env python3
"""
Production Testing Suite for AI Prediction Platform
Comprehensive testing of all API endpoints, authentication, and functionality
"""

import requests
import json
import os
import time

BASE_URL = "http://localhost:5000"
DEMO_API_KEY = "demo-api-key-12345"

def test_health_check():
    """Test basic health endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Health check passed: {data['status']}")
        return True
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
        return False

def test_user_registration():
    """Test user registration"""
    print("🔍 Testing user registration...")
    
    # Create a unique user
    timestamp = int(time.time())
    user_data = {
        "username": f"testuser_{timestamp}",
        "email": f"test_{timestamp}@example.com",
        "password": "securepass123"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/auth/register",
        headers={"Content-Type": "application/json"},
        json=user_data
    )
    
    if response.status_code == 201:
        data = response.json()
        print(f"   ✅ User registration successful: {data['user']['username']}")
        print(f"   🔑 API Key: {data['api_key'][:8]}...")
        print(f"   🎫 JWT Token: {data['token'][:20]}...")
        return data['api_key'], data['token']
    else:
        print(f"   ❌ User registration failed: {response.status_code}")
        print(f"   📝 Response: {response.text}")
        return None, None

def test_prediction_endpoints(api_key):
    """Test all prediction endpoints"""
    print("🔍 Testing prediction endpoints...")
    
    endpoints_tests = [
        {
            "name": "Lead Scoring",
            "endpoint": "/api/predict/lead-score",
            "payload": {
                "company_size": "51-200",
                "budget": 150000,
                "industry_score": 8.5,
                "engagement_score": 7.2,
                "demographic_score": 6.8,
                "behavioral_score": 8.1,
                "source_score": 7.5
            }
        },
        {
            "name": "Churn Prediction",
            "endpoint": "/api/predict/churn",
            "payload": {
                "tenure": 24,
                "monthly_charges": 85.5,
                "total_charges": 2052.0,
                "contract_type": "One year",
                "payment_method": "Credit card",
                "internet_service": "Fiber optic",
                "support_tickets": 2
            }
        },
        {
            "name": "Sales Forecast",
            "endpoint": "/api/predict/sales-forecast",
            "payload": {
                "historical_sales": 125000,
                "seasonality": 1.2,
                "marketing_spend": 25000,
                "economic_indicators": 1.05,
                "product_category": "Electronics"
            }
        },
        {
            "name": "NLP Analysis",
            "endpoint": "/api/predict/nlp",
            "payload": {
                "text": "This is an excellent product with outstanding customer service. I am very satisfied with my purchase and would definitely recommend it to others."
            }
        }
    ]
    
    results = {}
    
    for test in endpoints_tests:
        print(f"   Testing {test['name']}...")
        
        response = requests.post(
            f"{BASE_URL}{test['endpoint']}",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key
            },
            json=test['payload']
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"      ✅ {test['name']} successful")
            print(f"      ⏱️ Processing time: {data.get('processing_time', 'N/A'):.3f}s")
            if 'prediction' in data:
                prediction = data['prediction']
                if isinstance(prediction, dict):
                    # Show key prediction values
                    for key, value in list(prediction.items())[:3]:
                        if isinstance(value, (int, float)):
                            print(f"      📊 {key}: {value:.4f}")
                        else:
                            print(f"      📊 {key}: {value}")
            results[test['name']] = True
        else:
            print(f"      ❌ {test['name']} failed: {response.status_code}")
            print(f"      📝 Error: {response.text}")
            results[test['name']] = False
    
    return results

def test_industry_endpoints(api_key):
    """Test industry-specific endpoints"""
    print("🔍 Testing industry-specific endpoints...")
    
    industry_tests = [
        {
            "name": "Healthcare Churn",
            "endpoint": "/api/industry/healthcare/churn",
            "payload": {
                "patient_age": 45,
                "visit_frequency": 3,
                "insurance_type": "Premium",
                "treatment_cost": 1200,
                "satisfaction_score": 8.5
            }
        },
        {
            "name": "Finance Lead Score",
            "endpoint": "/api/industry/finance/lead-score",
            "payload": {
                "income_level": "High",
                "credit_score": 750,
                "investment_experience": "Experienced",
                "risk_tolerance": "Moderate"
            }
        },
        {
            "name": "Retail Price Optimization",
            "endpoint": "/api/industry/retail/price-optimization",
            "payload": {
                "product_category": "Electronics",
                "competitor_price": 299.99,
                "inventory_level": 150,
                "seasonal_demand": "High"
            }
        }
    ]
    
    results = {}
    
    for test in industry_tests:
        print(f"   Testing {test['name']}...")
        
        response = requests.post(
            f"{BASE_URL}{test['endpoint']}",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key
            },
            json=test['payload']
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"      ✅ {test['name']} successful")
            print(f"      ⏱️ Processing time: {data.get('processing_time', 'N/A'):.3f}s")
            results[test['name']] = True
        else:
            print(f"      ❌ {test['name']} failed: {response.status_code}")
            results[test['name']] = False
    
    return results

def test_file_upload(api_key):
    """Test file upload functionality"""
    print("🔍 Testing file upload...")
    
    # Test with our sample data
    file_path = "uploads/sample_lead_scoring_data.csv"
    
    if not os.path.exists(file_path):
        print(f"      ❌ Sample file not found: {file_path}")
        return False
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'model_type': 'lead_score'}
        
        response = requests.post(
            f"{BASE_URL}/api/upload/",
            headers={"X-API-Key": api_key},
            files=files,
            data=data
        )
    
    if response.status_code == 201:
        result = response.json()
        print(f"      ✅ File upload successful")
        print(f"      📁 Upload ID: {result.get('upload_id')}")
        print(f"      📊 Total rows: {result.get('total_rows')}")
        print(f"      🔍 Quality score: {result.get('data_quality_score'):.2f}")
        return True
    else:
        print(f"      ❌ File upload failed: {response.status_code}")
        print(f"      📝 Error: {response.text}")
        return False

def test_authentication_methods():
    """Test different authentication methods"""
    print("🔍 Testing authentication methods...")
    
    # Test API key authentication
    response = requests.post(
        f"{BASE_URL}/api/predict/lead-score",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": DEMO_API_KEY
        },
        json={
            "company_size": "11-50",
            "budget": 50000,
            "industry_score": 7.0,
            "engagement_score": 6.5,
            "demographic_score": 5.8,
            "behavioral_score": 7.2,
            "source_score": 6.0
        }
    )
    
    if response.status_code == 200:
        print("      ✅ API Key authentication successful")
    else:
        print(f"      ❌ API Key authentication failed: {response.status_code}")
    
    # Test missing authentication
    response = requests.post(
        f"{BASE_URL}/api/predict/lead-score",
        headers={"Content-Type": "application/json"},
        json={"test": "data"}
    )
    
    if response.status_code == 401:
        print("      ✅ Missing authentication properly rejected")
        return True
    else:
        print(f"      ❌ Missing authentication not handled: {response.status_code}")
        return False

def main():
    """Run comprehensive production tests"""
    print("🚀 Starting Production Testing Suite for AI Prediction Platform")
    print("=" * 70)
    print()
    
    # Test results tracking
    all_tests = {}
    
    # Basic functionality tests
    all_tests['health_check'] = test_health_check()
    print()
    
    # User registration
    api_key, jwt_token = test_user_registration()
    all_tests['user_registration'] = api_key is not None
    print()
    
    # Use demo API key if registration failed
    if not api_key:
        api_key = DEMO_API_KEY
        print(f"🔄 Using demo API key: {api_key}")
        print()
    
    # Authentication tests
    all_tests['authentication'] = test_authentication_methods()
    print()
    
    # Core prediction endpoints
    prediction_results = test_prediction_endpoints(api_key)
    all_tests.update(prediction_results)
    print()
    
    # Industry-specific endpoints
    industry_results = test_industry_endpoints(api_key)
    all_tests.update(industry_results)
    print()
    
    # File upload test
    all_tests['file_upload'] = test_file_upload(api_key)
    print()
    
    # Summary
    print("=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for result in all_tests.values() if result)
    total = len(all_tests)
    
    print(f"✅ Passed: {passed}/{total} tests")
    print()
    
    print("📋 Detailed Results:")
    for test_name, result in all_tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print()
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - APPLICATION IS PRODUCTION READY! 🎉")
    else:
        print(f"⚠️ {total - passed} tests failed - review issues before production deployment")
    
    print()
    print("🔗 Application URLs:")
    print(f"   • Main Dashboard: {BASE_URL}/")
    print(f"   • API Documentation: {BASE_URL}/api-docs")
    print(f"   • Health Check: {BASE_URL}/health")
    print(f"   • Admin Panel: {BASE_URL}/admin/")
    print()
    print("🔑 Demo Credentials:")
    print(f"   • Username: admin")
    print(f"   • Password: admin123")
    print(f"   • API Key: {DEMO_API_KEY}")

if __name__ == "__main__":
    main()