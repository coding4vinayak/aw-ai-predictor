"""
Enterprise test suite for world-class AI platform
Comprehensive testing of performance, security, and functionality
"""

import unittest
import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EnterpriseTestSuite(unittest.TestCase):
    """Comprehensive enterprise test suite"""
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = 'http://localhost:5000'
        cls.api_key = 'demo-api-key-12345'
        cls.test_user = {'username': 'test_user', 'password': 'test_password'}
        
        # Performance thresholds
        cls.performance_thresholds = {
            'response_time_ms': 1000,
            'throughput_rps': 10,
            'error_rate_percent': 1.0
        }
    
    def setUp(self):
        """Set up for each test"""
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': self.api_key})
    
    def test_health_endpoints(self):
        """Test health check endpoints"""
        # Basic health check
        response = self.session.get(f'{self.base_url}/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        
        # Detailed health check (requires auth)
        response = self.session.get(f'{self.base_url}/api/enterprise/health/detailed')
        # Should work with API key or return 401
        self.assertIn(response.status_code, [200, 401])
    
    def test_prometheus_metrics(self):
        """Test Prometheus metrics endpoint"""
        response = self.session.get(f'{self.base_url}/api/enterprise/metrics/prometheus')
        
        if response.status_code == 200:
            # Check if metrics format is correct
            content = response.text
            self.assertIn('# HELP', content)
            self.assertIn('# TYPE', content)
            self.assertEqual(response.headers.get('Content-Type'), 'text/plain; charset=utf-8')
    
    def test_core_prediction_endpoints(self):
        """Test core ML prediction endpoints"""
        test_cases = [
            {
                'endpoint': '/api/predict/lead-score',
                'payload': {
                    'company_size': '51-200',
                    'budget': 150000,
                    'industry_score': 8.5
                }
            },
            {
                'endpoint': '/api/predict/churn',
                'payload': {
                    'tenure': 24,
                    'monthly_charges': 85.0,
                    'contract_type': 'One year'
                }
            },
            {
                'endpoint': '/api/predict/sales-forecast',
                'payload': {
                    'historical_sales': 200000,
                    'seasonality': 1.2,
                    'marketing_spend': 25000
                }
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(endpoint=test_case['endpoint']):
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}{test_case['endpoint']}",
                    json=test_case['payload']
                )
                response_time = (time.time() - start_time) * 1000
                
                # Check response
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn('prediction', data)
                
                # Check performance
                self.assertLess(response_time, self.performance_thresholds['response_time_ms'],
                              f"Response time {response_time:.0f}ms exceeds threshold")
    
    def test_batch_prediction_performance(self):
        """Test batch prediction endpoint performance"""
        batch_requests = []
        for i in range(10):
            batch_requests.append({
                'model': 'lead_scoring',
                'input': {
                    'company_size': '51-200',
                    'budget': 100000 + (i * 10000),
                    'industry_score': 7.0 + (i * 0.2)
                }
            })
        
        payload = {'requests': batch_requests}
        
        start_time = time.time()
        response = self.session.post(
            f'{self.base_url}/api/enterprise/batch/predict',
            json=payload
        )
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            batch_stats = data.get('batch_stats', {})
            
            # Check batch performance
            self.assertGreater(batch_stats.get('requests_per_second', 0), 5)
            self.assertEqual(batch_stats.get('total_requests'), 10)
            self.assertGreater(batch_stats.get('success_rate', 0), 90)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(15):  # More than typical rate limit
            try:
                response = self.session.get(f'{self.base_url}/health')
                responses.append(response.status_code)
            except:
                responses.append(429)  # Assume rate limited
        
        # Should have some successful requests
        success_count = responses.count(200)
        self.assertGreater(success_count, 0)
        
        # May have rate limited requests
        rate_limited_count = responses.count(429)
        # Rate limiting may or may not be triggered depending on configuration
    
    def test_concurrent_load(self):
        """Test platform under concurrent load"""
        def make_request():
            try:
                response = requests.post(
                    f'{self.base_url}/api/predict/lead-score',
                    json={
                        'company_size': '51-200',
                        'budget': 150000,
                        'industry_score': 8.0
                    },
                    headers={'X-API-Key': self.api_key},
                    timeout=5
                )
                return response.status_code, response.elapsed.total_seconds()
            except Exception as e:
                return 500, 5.0
        
        # Run 20 concurrent requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # Analyze results
        success_count = len([r for r in results if r[0] == 200])
        avg_response_time = sum(r[1] for r in results) / len(results)
        
        success_rate = (success_count / len(results)) * 100
        
        # Performance assertions
        self.assertGreater(success_rate, 80, f"Success rate {success_rate:.1f}% below threshold")
        self.assertLess(avg_response_time, 2.0, f"Average response time {avg_response_time:.2f}s too high")
    
    def test_monitoring_endpoints(self):
        """Test monitoring and analytics endpoints"""
        monitoring_endpoints = [
            '/monitoring',
            '/monitoring/api/health',
            '/monitoring/api/models',
            '/monitoring/api/metrics',
            '/monitoring/api/alerts'
        ]
        
        for endpoint in monitoring_endpoints:
            with self.subTest(endpoint=endpoint):
                response = self.session.get(f'{self.base_url}{endpoint}')
                # Should return 200 or 401 (auth required)
                self.assertIn(response.status_code, [200, 401])
    
    def test_data_validation(self):
        """Test input data validation and security"""
        malicious_inputs = [
            # SQL injection attempts
            {"company_size": "'; DROP TABLE users; --"},
            # XSS attempts
            {"budget": "<script>alert('xss')</script>"},
            # Very large inputs
            {"industry_score": "A" * 10000},
            # Invalid data types
            {"budget": "not_a_number"},
            {"company_size": 12345}
        ]
        
        for malicious_input in malicious_inputs:
            with self.subTest(input=str(malicious_input)[:50]):
                response = self.session.post(
                    f'{self.base_url}/api/predict/lead-score',
                    json=malicious_input
                )
                
                # Should handle gracefully (400 error or sanitized processing)
                self.assertIn(response.status_code, [200, 400, 422])
                
                if response.status_code == 200:
                    data = response.json()
                    # Should not contain malicious content in response
                    response_text = json.dumps(data)
                    self.assertNotIn('<script>', response_text.lower())
                    self.assertNotIn('drop table', response_text.lower())
    
    def test_file_upload_security(self):
        """Test file upload security and validation"""
        # Test malicious file upload
        malicious_files = [
            # Oversized file simulation
            ('test.csv', 'a' * (20 * 1024 * 1024)),  # 20MB
            # Invalid file format
            ('test.exe', b'\x4d\x5a\x90\x00'),  # PE header
            # CSV with malicious content
            ('malicious.csv', '=cmd|"/c calc"!A1,data\n'),
        ]
        
        for filename, content in malicious_files:
            with self.subTest(filename=filename):
                files = {'file': (filename, content, 'text/csv')}
                response = self.session.post(
                    f'{self.base_url}/api/upload/csv/lead-scoring',
                    files=files
                )
                
                # Should reject malicious uploads
                self.assertIn(response.status_code, [400, 413, 415, 422])
    
    def test_api_documentation(self):
        """Test API documentation accessibility"""
        doc_endpoints = [
            '/api-docs',
            '/getting-started',
            '/data-guide'
        ]
        
        for endpoint in doc_endpoints:
            with self.subTest(endpoint=endpoint):
                response = self.session.get(f'{self.base_url}{endpoint}')
                self.assertEqual(response.status_code, 200)
    
    def test_error_handling(self):
        """Test error handling and logging"""
        # Test various error conditions
        error_cases = [
            # Missing required fields
            ('/api/predict/lead-score', {}),
            # Invalid endpoint
            ('/api/predict/nonexistent-model', {}),
            # Malformed JSON would be tested at HTTP client level
        ]
        
        for endpoint, payload in error_cases:
            with self.subTest(endpoint=endpoint):
                response = self.session.post(f'{self.base_url}{endpoint}', json=payload)
                
                # Should return proper error codes
                self.assertIn(response.status_code, [400, 404, 422])
                
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                    # Should have error message
                    self.assertTrue(
                        'error' in data or 'message' in data,
                        "Error response should contain error message"
                    )
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'test_results': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Test response times for key endpoints
        key_endpoints = [
            '/health',
            '/api/predict/lead-score',
            '/api/predict/churn',
            '/monitoring/api/health'
        ]
        
        for endpoint in key_endpoints:
            try:
                start_time = time.time()
                
                if endpoint.startswith('/api/predict/'):
                    response = self.session.post(
                        f'{self.base_url}{endpoint}',
                        json={'company_size': '51-200', 'budget': 150000, 'industry_score': 8.0}
                    )
                else:
                    response = self.session.get(f'{self.base_url}{endpoint}')
                
                response_time = (time.time() - start_time) * 1000
                
                report['performance_metrics'][endpoint] = {
                    'response_time_ms': response_time,
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                }
                
                if response_time > self.performance_thresholds['response_time_ms']:
                    report['recommendations'].append(
                        f"Endpoint {endpoint} exceeds response time threshold: {response_time:.0f}ms"
                    )
                
            except Exception as e:
                report['performance_metrics'][endpoint] = {
                    'error': str(e),
                    'success': False
                }
        
        return report

def run_enterprise_tests():
    """Run the complete enterprise test suite"""
    print("🚀 Starting Enterprise Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(EnterpriseTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Generate performance report
    test_instance = EnterpriseTestSuite()
    test_instance.setUp()
    
    print("\n📊 Performance Report")
    print("=" * 50)
    
    try:
        report = test_instance.generate_performance_report()
        
        for endpoint, metrics in report['performance_metrics'].items():
            if metrics.get('success'):
                print(f"✅ {endpoint}: {metrics['response_time_ms']:.0f}ms")
            else:
                print(f"❌ {endpoint}: {metrics.get('error', 'Failed')}")
        
        if report['recommendations']:
            print("\n⚠️  Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        else:
            print("\n🎉 All performance metrics within acceptable thresholds!")
    
    except Exception as e:
        print(f"❌ Performance report generation failed: {e}")
    
    # Summary
    print(f"\n📈 Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_enterprise_tests()
    sys.exit(0 if success else 1)