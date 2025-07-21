"""
Enterprise-grade monitoring and metrics collection for AI platform
Real-time tracking of model performance, API health, and business KPIs
"""

import time
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
import threading
from flask import request, g
from functools import wraps
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# In-memory metrics store (replace with Redis/Prometheus in production)
class MetricsStore:
    def __init__(self, max_datapoints=10000):
        self.max_datapoints = max_datapoints
        self.metrics = defaultdict(lambda: deque(maxlen=max_datapoints))
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = defaultdict(float)
        self.lock = Lock()
        
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value with timestamp"""
        with self.lock:
            timestamp = time.time()
            metric_data = {
                'timestamp': timestamp,
                'value': value,
                'labels': labels or {}
            }
            self.metrics[name].append(metric_data)
    
    def increment_counter(self, name: str, amount: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True) if labels else ''}"
            self.counters[key] += amount
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        with self.lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True) if labels else ''}"
            self.gauges[key] = value
    
    def add_histogram_sample(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add a sample to histogram"""
        with self.lock:
            key = f"{name}:{json.dumps(labels, sort_keys=True) if labels else ''}"
            self.histograms[key].append(value)
            # Keep only last 1000 samples
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def get_metrics_summary(self, time_window_minutes: int = 60):
        """Get metrics summary for dashboard"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        summary = {}
        
        with self.lock:
            # Process time-series metrics
            for name, datapoints in self.metrics.items():
                recent_points = [p for p in datapoints if p['timestamp'] > cutoff_time]
                if recent_points:
                    values = [p['value'] for p in recent_points]
                    summary[name] = {
                        'count': len(values),
                        'avg': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'p95': np.percentile(values, 95) if len(values) > 1 else values[0],
                        'latest': values[-1] if values else 0
                    }
            
            # Add counters
            summary['counters'] = dict(self.counters)
            
            # Add gauges
            summary['gauges'] = dict(self.gauges)
            
            # Add histogram statistics
            for name, samples in self.histograms.items():
                if samples:
                    summary[f"histogram_{name}"] = {
                        'count': len(samples),
                        'avg': np.mean(samples),
                        'p50': np.percentile(samples, 50),
                        'p95': np.percentile(samples, 95),
                        'p99': np.percentile(samples, 99)
                    }
        
        return summary

# Global metrics store
metrics_store = MetricsStore()

@dataclass
class ModelMetrics:
    """Model-specific performance metrics"""
    model_name: str
    predictions_count: int = 0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    error_rate: float = 0.0
    accuracy_trend: List[float] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.accuracy_trend is None:
            self.accuracy_trend = []
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

class ModelMonitor:
    """Advanced model performance monitoring"""
    
    def __init__(self):
        self.model_metrics = {}
        self.data_drift_detector = DataDriftDetector()
        self.concept_drift_detector = ConceptDriftDetector()
        
    def log_prediction(self, model_name: str, confidence: float, processing_time: float, 
                      input_features: Dict, prediction: Any, status: str = 'success'):
        """Log a model prediction with comprehensive metrics"""
        
        # Update model metrics
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(model_name=model_name)
        
        model_metric = self.model_metrics[model_name]
        model_metric.predictions_count += 1
        model_metric.last_updated = datetime.utcnow()
        
        if status == 'success':
            # Update running averages
            alpha = 0.1  # Smoothing factor
            model_metric.avg_confidence = (
                alpha * confidence + (1 - alpha) * model_metric.avg_confidence
            )
            model_metric.avg_processing_time = (
                alpha * processing_time + (1 - alpha) * model_metric.avg_processing_time
            )
        else:
            # Track error rate
            error_count = getattr(model_metric, '_error_count', 0) + 1
            setattr(model_metric, '_error_count', error_count)
            model_metric.error_rate = error_count / model_metric.predictions_count
        
        # Record metrics in store
        metrics_store.record_metric(f'model_confidence_{model_name}', confidence)
        metrics_store.record_metric(f'model_processing_time_{model_name}', processing_time)
        metrics_store.increment_counter(f'model_predictions_{model_name}', labels={'status': status})
        
        # Check for data drift
        self.data_drift_detector.check_drift(model_name, input_features)
        
        logging.debug(f"Logged prediction for {model_name}: confidence={confidence:.3f}, time={processing_time:.3f}s")
    
    def get_model_health_score(self, model_name: str) -> float:
        """Calculate overall model health score (0-100)"""
        if model_name not in self.model_metrics:
            return 0.0
        
        metric = self.model_metrics[model_name]
        
        # Health factors
        confidence_score = min(100, metric.avg_confidence * 100)
        performance_score = max(0, 100 - (metric.avg_processing_time * 10))  # Penalize slow models
        error_score = max(0, 100 - (metric.error_rate * 100))
        
        # Weighted average
        health_score = (confidence_score * 0.4 + performance_score * 0.3 + error_score * 0.3)
        return min(100, health_score)
    
    def get_dashboard_data(self):
        """Get comprehensive dashboard data"""
        dashboard_data = {
            'models': {},
            'system_metrics': metrics_store.get_metrics_summary(),
            'alerts': [],
            'drift_status': {}
        }
        
        for model_name, metric in self.model_metrics.items():
            health_score = self.get_model_health_score(model_name)
            
            dashboard_data['models'][model_name] = {
                'predictions_count': metric.predictions_count,
                'avg_confidence': round(metric.avg_confidence, 3),
                'avg_processing_time': round(metric.avg_processing_time, 3),
                'error_rate': round(metric.error_rate * 100, 2),
                'health_score': round(health_score, 1),
                'last_updated': metric.last_updated.isoformat(),
                'status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical'
            }
            
            # Generate alerts for poor performance
            if health_score < 70:
                dashboard_data['alerts'].append({
                    'severity': 'high' if health_score < 50 else 'medium',
                    'message': f"Model {model_name} health score is {health_score:.1f}%",
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return dashboard_data

class DataDriftDetector:
    """Detect data drift in model inputs"""
    
    def __init__(self, max_samples=1000):
        self.reference_distributions = {}
        self.current_distributions = {}
        self.max_samples = max_samples
    
    def check_drift(self, model_name: str, features: Dict):
        """Check for data drift in incoming features"""
        # Store reference distribution (first N samples)
        if model_name not in self.reference_distributions:
            self.reference_distributions[model_name] = defaultdict(list)
            self.current_distributions[model_name] = defaultdict(list)
        
        ref_dist = self.reference_distributions[model_name]
        curr_dist = self.current_distributions[model_name]
        
        # Collect feature values
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                if len(ref_dist[feature]) < self.max_samples // 2:
                    ref_dist[feature].append(value)
                else:
                    curr_dist[feature].append(value)
                    if len(curr_dist[feature]) > self.max_samples // 2:
                        curr_dist[feature].pop(0)  # Keep recent samples
                
                # Check for drift if we have enough samples
                if len(ref_dist[feature]) > 50 and len(curr_dist[feature]) > 50:
                    drift_score = self._calculate_drift_score(ref_dist[feature], curr_dist[feature])
                    if drift_score > 0.1:  # Threshold for significant drift
                        logging.warning(f"Data drift detected for {model_name}, feature {feature}: score={drift_score:.3f}")
                        metrics_store.record_metric(f'data_drift_{model_name}_{feature}', drift_score)
    
    def _calculate_drift_score(self, reference: List[float], current: List[float]) -> float:
        """Calculate drift score using statistical distance"""
        try:
            ref_mean, ref_std = np.mean(reference), np.std(reference)
            curr_mean, curr_std = np.mean(current), np.std(current)
            
            # Simple drift score based on mean and std differences
            mean_drift = abs(ref_mean - curr_mean) / (ref_std + 1e-8)
            std_drift = abs(ref_std - curr_std) / (ref_std + 1e-8)
            
            return (mean_drift + std_drift) / 2
        except:
            return 0.0

class ConceptDriftDetector:
    """Detect concept drift in model predictions"""
    
    def __init__(self):
        self.accuracy_windows = defaultdict(lambda: deque(maxlen=100))
    
    def log_accuracy(self, model_name: str, accuracy: float):
        """Log model accuracy for drift detection"""
        self.accuracy_windows[model_name].append(accuracy)
        
        # Check for concept drift
        if len(self.accuracy_windows[model_name]) >= 50:
            recent_accuracy = np.mean(list(self.accuracy_windows[model_name])[-25:])
            older_accuracy = np.mean(list(self.accuracy_windows[model_name])[:25])
            
            if abs(recent_accuracy - older_accuracy) > 0.05:  # 5% accuracy drop
                logging.warning(f"Concept drift detected for {model_name}: accuracy drop of {(older_accuracy - recent_accuracy)*100:.1f}%")

# Global monitor instance
model_monitor = ModelMonitor()

def track_api_performance(f):
    """Decorator to track API endpoint performance"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        endpoint = request.endpoint or 'unknown'
        method = request.method
        
        try:
            # Execute the function
            result = f(*args, **kwargs)
            
            # Record success metrics
            processing_time = time.time() - start_time
            metrics_store.record_metric('api_response_time', processing_time, 
                                      {'endpoint': endpoint, 'method': method})
            metrics_store.increment_counter('api_requests_total', 
                                          labels={'endpoint': endpoint, 'method': method, 'status': 'success'})
            
            return result
            
        except Exception as e:
            # Record error metrics
            processing_time = time.time() - start_time
            metrics_store.increment_counter('api_requests_total',
                                          labels={'endpoint': endpoint, 'method': method, 'status': 'error'})
            metrics_store.record_metric('api_error_rate', 1.0, {'endpoint': endpoint})
            
            logging.error(f"API error in {endpoint}: {str(e)}")
            raise
    
    return decorated_function

def track_prediction_performance(model_name: str):
    """Decorator to track ML model prediction performance"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Execute prediction
                result = f(*args, **kwargs)
                processing_time = time.time() - start_time
                
                # Extract metrics from result
                confidence = 0.0
                if isinstance(result, dict):
                    confidence = result.get('confidence', 0.0)
                    if isinstance(confidence, (list, np.ndarray)):
                        confidence = float(np.mean(confidence))
                    else:
                        confidence = float(confidence)
                
                # Log to model monitor
                input_features = kwargs.get('data', {}) if kwargs else {}
                model_monitor.log_prediction(
                    model_name=model_name,
                    confidence=confidence,
                    processing_time=processing_time,
                    input_features=input_features,
                    prediction=result,
                    status='success'
                )
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                model_monitor.log_prediction(
                    model_name=model_name,
                    confidence=0.0,
                    processing_time=processing_time,
                    input_features={},
                    prediction=None,
                    status='error'
                )
                raise
        
        return decorated_function
    return decorator

# Utility functions for dashboard integration
def get_system_health():
    """Get overall system health metrics"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': metrics_store.get_metrics_summary(),
        'models': model_monitor.get_dashboard_data()
    }

def reset_metrics():
    """Reset all metrics (for testing)"""
    global metrics_store, model_monitor
    metrics_store = MetricsStore()
    model_monitor = ModelMonitor()

# Background thread for metric aggregation
def start_metrics_aggregator():
    """Start background thread for metric aggregation"""
    def aggregate_metrics():
        while True:
            try:
                # Perform periodic metric aggregations
                time.sleep(60)  # Run every minute
                
                # Calculate system-wide metrics
                summary = metrics_store.get_metrics_summary()
                
                # Log key metrics
                logging.info(f"System metrics: {json.dumps(summary, indent=2)}")
                
            except Exception as e:
                logging.error(f"Error in metrics aggregator: {e}")
    
    thread = threading.Thread(target=aggregate_metrics, daemon=True)
    thread.start()
    logging.info("Started metrics aggregator thread")

# Initialize metrics aggregation on module import
start_metrics_aggregator()