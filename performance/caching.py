"""
Enterprise-grade caching system for AI platform
Intelligent caching of predictions, models, and API responses
"""

import time
import json
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
from functools import wraps
import threading
from collections import OrderedDict

class LRUCache:
    """Thread-safe LRU Cache implementation"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value, timestamp, ttl = self.cache.pop(key)
                
                # Check if expired
                if ttl and time.time() > timestamp + ttl:
                    self.stats['misses'] += 1
                    return None
                
                self.cache[key] = (value, timestamp, ttl)
                self.stats['hits'] += 1
                return value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict if at capacity
            while len(self.cache) >= self.maxsize:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
            
            self.cache[key] = (value, time.time(), ttl)
    
    def delete(self, key: str):
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get_stats(self) -> Dict:
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self.stats,
                'size': len(self.cache),
                'hit_rate': round(hit_rate, 2),
                'total_requests': total_requests
            }

class PredictionCache:
    """Specialized cache for ML predictions"""
    
    def __init__(self, maxsize: int = 5000, default_ttl: int = 3600):
        self.cache = LRUCache(maxsize)
        self.default_ttl = default_ttl
        
        # Model-specific cache settings
        self.model_settings = {
            'lead_scoring': {'ttl': 1800, 'enabled': True},      # 30 min
            'churn_prediction': {'ttl': 3600, 'enabled': True},  # 1 hour
            'sales_forecast': {'ttl': 7200, 'enabled': True},    # 2 hours
            'sentiment': {'ttl': 300, 'enabled': True},          # 5 min
            'nlp': {'ttl': 300, 'enabled': True}                 # 5 min
        }
    
    def _generate_cache_key(self, model_name: str, input_data: Dict) -> str:
        """Generate deterministic cache key from model and input"""
        # Sort input data for consistent hashing
        sorted_input = json.dumps(input_data, sort_keys=True, default=str)
        content = f"{model_name}:{sorted_input}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_prediction(self, model_name: str, input_data: Dict) -> Optional[Dict]:
        """Get cached prediction if available"""
        if not self.model_settings.get(model_name, {}).get('enabled', False):
            return None
        
        cache_key = self._generate_cache_key(model_name, input_data)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logging.debug(f"Cache hit for {model_name}: {cache_key[:8]}...")
            return cached_result
        
        return None
    
    def set_prediction(self, model_name: str, input_data: Dict, prediction: Dict):
        """Cache prediction result"""
        if not self.model_settings.get(model_name, {}).get('enabled', False):
            return
        
        cache_key = self._generate_cache_key(model_name, input_data)
        ttl = self.model_settings.get(model_name, {}).get('ttl', self.default_ttl)
        
        # Add cache metadata
        cached_data = {
            **prediction,
            '_cached_at': time.time(),
            '_cache_key': cache_key[:8]
        }
        
        self.cache.set(cache_key, cached_data, ttl)
        logging.debug(f"Cached prediction for {model_name}: {cache_key[:8]}...")
    
    def invalidate_model(self, model_name: str):
        """Invalidate all cached predictions for a model"""
        # This would require key prefix scanning in a full Redis implementation
        logging.info(f"Model cache invalidation requested for {model_name}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        base_stats = self.cache.get_stats()
        return {
            **base_stats,
            'model_settings': self.model_settings,
            'cache_type': 'prediction_cache'
        }

class ModelCache:
    """Cache for loaded ML models and preprocessors"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.lock = threading.RLock()
    
    def get_model(self, model_name: str, version: str = 'latest') -> Optional[Any]:
        """Get cached model"""
        key = f"{model_name}:{version}"
        
        with self.lock:
            if key in self.models:
                self.model_metadata[key]['last_accessed'] = time.time()
                self.model_metadata[key]['access_count'] += 1
                return self.models[key]
        
        return None
    
    def set_model(self, model_name: str, model_obj: Any, version: str = 'latest', 
                  metadata: Optional[Dict] = None):
        """Cache model object"""
        key = f"{model_name}:{version}"
        
        with self.lock:
            self.models[key] = model_obj
            self.model_metadata[key] = {
                'loaded_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'size_mb': self._estimate_size(model_obj),
                'metadata': metadata or {}
            }
    
    def _estimate_size(self, obj: Any) -> float:
        """Estimate object size in MB"""
        try:
            return len(pickle.dumps(obj)) / (1024 * 1024)
        except:
            return 0.0
    
    def get_stats(self) -> Dict:
        """Get model cache statistics"""
        with self.lock:
            total_size = sum(meta['size_mb'] for meta in self.model_metadata.values())
            total_access = sum(meta['access_count'] for meta in self.model_metadata.values())
            
            return {
                'cached_models': len(self.models),
                'total_size_mb': round(total_size, 2),
                'total_accesses': total_access,
                'models': {
                    key: {
                        'size_mb': meta['size_mb'],
                        'access_count': meta['access_count'],
                        'loaded_at': datetime.fromtimestamp(meta['loaded_at']).isoformat(),
                        'last_accessed': datetime.fromtimestamp(meta['last_accessed']).isoformat()
                    }
                    for key, meta in self.model_metadata.items()
                }
            }

# Global cache instances
prediction_cache = PredictionCache()
model_cache = ModelCache()

def cache_prediction(model_name: str, ttl: Optional[int] = None):
    """
    Decorator for caching ML predictions
    
    Args:
        model_name: Name of the ML model
        ttl: Time to live in seconds (optional)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract input data for caching
            input_data = {}
            if args:
                input_data = args[0] if isinstance(args[0], dict) else {}
            input_data.update(kwargs.get('data', {}))
            
            # Try to get from cache
            cached_result = prediction_cache.get_prediction(model_name, input_data)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if isinstance(result, dict) and 'error' not in result:
                prediction_cache.set_prediction(model_name, input_data, result)
            
            return result
        
        return wrapper
    return decorator

def cache_model_loading(model_name: str, version: str = 'latest'):
    """
    Decorator for caching loaded models
    
    Args:
        model_name: Name of the model
        version: Model version
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_model = model_cache.get_model(model_name, version)
            if cached_model is not None:
                return cached_model
            
            # Load model and cache it
            model_obj = func(*args, **kwargs)
            
            if model_obj is not None:
                metadata = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                model_cache.set_model(model_name, model_obj, version, metadata)
            
            return model_obj
        
        return wrapper
    return decorator

class ResponseCache:
    """HTTP response caching for API endpoints"""
    
    def __init__(self, maxsize: int = 1000):
        self.cache = LRUCache(maxsize)
    
    def get_response(self, cache_key: str) -> Optional[tuple]:
        """Get cached response (data, status_code, headers)"""
        return self.cache.get(cache_key)
    
    def set_response(self, cache_key: str, data: Any, status_code: int, 
                    headers: Dict, ttl: int = 300):
        """Cache HTTP response"""
        response_data = (data, status_code, dict(headers))
        self.cache.set(cache_key, response_data, ttl)
    
    def generate_key(self, endpoint: str, params: Dict, user_id: str = None) -> str:
        """Generate cache key for HTTP request"""
        key_data = {
            'endpoint': endpoint,
            'params': params,
            'user': user_id or 'anonymous'
        }
        content = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

# Global response cache
response_cache = ResponseCache()

def cache_response(ttl: int = 300, vary_by_user: bool = True):
    """
    Decorator for caching HTTP responses
    
    Args:
        ttl: Time to live in seconds
        vary_by_user: Whether to vary cache by user ID
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            
            # Generate cache key
            user_id = kwargs.get('user_id') if vary_by_user else None
            cache_key = response_cache.generate_key(
                request.endpoint,
                dict(request.args),
                user_id
            )
            
            # Try to get from cache
            cached_response = response_cache.get_response(cache_key)
            if cached_response:
                data, status_code, headers = cached_response
                response = jsonify(data)
                response.status_code = status_code
                for key, value in headers.items():
                    response.headers[key] = value
                response.headers['X-Cache-Status'] = 'HIT'
                return response
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache successful responses
            if hasattr(result, 'status_code') and result.status_code == 200:
                data = result.get_json() if hasattr(result, 'get_json') else {}
                headers = dict(result.headers) if hasattr(result, 'headers') else {}
                
                response_cache.set_response(
                    cache_key, data, result.status_code, headers, ttl
                )
                
                result.headers['X-Cache-Status'] = 'MISS'
            
            return result
        
        return wrapper
    return decorator

def get_cache_stats() -> Dict:
    """Get comprehensive cache statistics"""
    return {
        'prediction_cache': prediction_cache.get_stats(),
        'model_cache': model_cache.get_stats(),
        'response_cache': response_cache.cache.get_stats(),
        'timestamp': datetime.utcnow().isoformat()
    }

def clear_all_caches():
    """Clear all cache instances"""
    prediction_cache.cache.clear()
    model_cache.models.clear()
    model_cache.model_metadata.clear()
    response_cache.cache.clear()
    logging.info("All caches cleared")

# Cache warming utilities
def warm_prediction_cache():
    """Pre-warm cache with common predictions"""
    common_requests = [
        ('lead_scoring', {'company_size': '51-200', 'budget': 100000, 'industry_score': 7.5}),
        ('churn_prediction', {'tenure': 12, 'monthly_charges': 70.0, 'contract_type': 'Month-to-month'}),
        ('sales_forecast', {'historical_sales': 150000, 'seasonality': 1.1, 'marketing_spend': 20000})
    ]
    
    for model_name, sample_data in common_requests:
        try:
            # This would call the actual prediction function in a real implementation
            logging.info(f"Cache warming for {model_name} - would execute prediction")
        except Exception as e:
            logging.error(f"Cache warming failed for {model_name}: {e}")

# Background cache maintenance
def start_cache_maintenance():
    """Start background thread for cache maintenance"""
    def maintenance_worker():
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Log cache statistics
                stats = get_cache_stats()
                logging.info(f"Cache stats: {json.dumps(stats, indent=2)}")
                
                # Warm cache during low traffic periods
                current_hour = datetime.now().hour
                if 2 <= current_hour <= 5:  # 2-5 AM
                    warm_prediction_cache()
                
            except Exception as e:
                logging.error(f"Cache maintenance error: {e}")
    
    thread = threading.Thread(target=maintenance_worker, daemon=True)
    thread.start()
    logging.info("Started cache maintenance thread")

# Initialize cache maintenance
start_cache_maintenance()