"""
Enterprise-grade rate limiting and security controls
Implements sliding window rate limiting with Redis backend support
"""

import time
import json
import logging
from collections import defaultdict, deque
from threading import Lock
from flask import request, jsonify
from functools import wraps
from datetime import datetime, timedelta
import hashlib
import hmac
from typing import Dict, Tuple, Optional

class RateLimiter:
    """
    Advanced rate limiter with multiple algorithms and security features
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        # In-memory fallback for development
        self.memory_store = defaultdict(lambda: deque())
        self.lock = Lock()
        
        # Rate limit configurations
        self.rate_limits = {
            'api_key': {'requests': 1000, 'window': 3600},  # 1000 per hour
            'ip': {'requests': 100, 'window': 3600},         # 100 per hour per IP
            'user': {'requests': 5000, 'window': 3600},      # 5000 per hour per user
            'endpoint': {'requests': 50, 'window': 60},      # 50 per minute per endpoint
            'ml_prediction': {'requests': 200, 'window': 3600}, # 200 ML predictions per hour
        }
        
        # Security thresholds
        self.security_limits = {
            'failed_auth': {'attempts': 5, 'window': 900},   # 5 failed attempts in 15 min
            'suspicious_patterns': {'requests': 20, 'window': 60}, # 20 rapid requests in 1 min
        }
    
    def _get_current_window_start(self, window_seconds: int) -> int:
        """Get the start of the current time window"""
        now = int(time.time())
        return now - (now % window_seconds)
    
    def _clean_old_entries(self, key: str, window_seconds: int):
        """Remove entries outside the current window"""
        cutoff = time.time() - window_seconds
        
        if self.redis:
            # Redis implementation with expiring keys
            pipeline = self.redis.pipeline()
            pipeline.zremrangebyscore(key, '-inf', cutoff)
            pipeline.expire(key, window_seconds)
            pipeline.execute()
        else:
            # Memory implementation
            if key in self.memory_store:
                while (self.memory_store[key] and 
                       self.memory_store[key][0] < cutoff):
                    self.memory_store[key].popleft()
    
    def _increment_counter(self, key: str, window_seconds: int) -> int:
        """Increment counter and return current count"""
        now = time.time()
        
        if self.redis:
            # Redis sliding window counter
            pipeline = self.redis.pipeline()
            pipeline.zadd(key, {str(now): now})
            pipeline.zcount(key, now - window_seconds, now)
            pipeline.expire(key, window_seconds)
            results = pipeline.execute()
            return results[1]
        else:
            # Memory implementation
            with self.lock:
                self.memory_store[key].append(now)
                self._clean_old_entries(key, window_seconds)
                return len(self.memory_store[key])
    
    def check_rate_limit(self, identifier: str, limit_type: str) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limits
        Returns: (allowed, info)
        """
        if limit_type not in self.rate_limits:
            return True, {'error': f'Unknown limit type: {limit_type}'}
        
        config = self.rate_limits[limit_type]
        key = f"rate_limit:{limit_type}:{identifier}"
        
        # Clean old entries and increment
        self._clean_old_entries(key, config['window'])
        current_count = self._increment_counter(key, config['window'])
        
        # Calculate remaining requests and reset time
        remaining = max(0, config['requests'] - current_count)
        reset_time = int(time.time()) + config['window']
        
        allowed = current_count <= config['requests']
        
        return allowed, {
            'limit': config['requests'],
            'remaining': remaining,
            'reset': reset_time,
            'current': current_count,
            'window': config['window']
        }
    
    def check_security_limits(self, identifier: str, event_type: str) -> Tuple[bool, Dict]:
        """Check security-related rate limits"""
        if event_type not in self.security_limits:
            return True, {}
        
        config = self.security_limits[event_type]
        key = f"security:{event_type}:{identifier}"
        
        self._clean_old_entries(key, config['window'])
        current_count = self._increment_counter(key, config['window'])
        
        allowed = current_count <= config['attempts']
        
        if not allowed:
            logging.warning(f"Security limit exceeded: {event_type} for {identifier} - {current_count} attempts")
        
        return allowed, {
            'limit': config['attempts'],
            'current': current_count,
            'blocked': not allowed
        }

# Global rate limiter instance
rate_limiter = RateLimiter()

def apply_rate_limit(limit_type: str, get_identifier=None):
    """
    Rate limiting decorator
    
    Args:
        limit_type: Type of rate limit to apply
        get_identifier: Function to extract identifier from request
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get identifier for rate limiting
            if get_identifier:
                identifier = get_identifier()
            else:
                # Default identifier extraction
                if limit_type == 'ip':
                    identifier = request.remote_addr
                elif limit_type == 'api_key':
                    identifier = request.headers.get('X-API-Key', 'anonymous')
                elif limit_type == 'user':
                    identifier = getattr(request, 'user_id', 'anonymous')
                else:
                    identifier = f"{request.remote_addr}:{request.endpoint}"
            
            # Check rate limit
            allowed, info = rate_limiter.check_rate_limit(identifier, limit_type)
            
            if not allowed:
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {info["limit"]} per {info["window"]} seconds',
                    'retry_after': info['reset'] - int(time.time()),
                    'limit': info['limit'],
                    'remaining': info['remaining'],
                    'reset': info['reset']
                })
                response.status_code = 429
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset'])
                response.headers['Retry-After'] = str(info['reset'] - int(time.time()))
                return response
            
            # Add rate limit headers to response
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset'])
            
            return response
        
        return decorated_function
    return decorator

def security_monitor(event_type: str):
    """Security monitoring decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            identifier = request.remote_addr
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Check for authentication failures
                if (hasattr(result, 'status_code') and 
                    result.status_code == 401 and 
                    event_type == 'authentication'):
                    
                    allowed, info = rate_limiter.check_security_limits(identifier, 'failed_auth')
                    if not allowed:
                        # Block IP for repeated auth failures
                        logging.warning(f"IP {identifier} blocked for repeated authentication failures")
                        return jsonify({
                            'error': 'Too many failed authentication attempts',
                            'message': 'Your IP has been temporarily blocked'
                        }), 429
                
                return result
                
            except Exception as e:
                logging.error(f"Security monitor error: {e}")
                raise
        
        return decorated_function
    return decorator

class SecurityValidator:
    """Advanced security validation and threat detection"""
    
    @staticmethod
    def validate_api_signature(payload: str, signature: str, secret: str) -> bool:
        """Validate HMAC signature for webhook security"""
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    @staticmethod
    def detect_sql_injection(input_string: str) -> bool:
        """Basic SQL injection pattern detection"""
        if not isinstance(input_string, str):
            return False
        
        sql_patterns = [
            r"union\s+select", r"drop\s+table", r"delete\s+from",
            r"insert\s+into", r"update\s+set", r"exec\s*\(",
            r"script\s*>", r"javascript:", r"on\w+\s*="
        ]
        
        input_lower = input_string.lower()
        for pattern in sql_patterns:
            if pattern in input_lower:
                return True
        return False
    
    @staticmethod
    def sanitize_input(input_data: any) -> any:
        """Sanitize input data for security"""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', ';']
            for char in dangerous_chars:
                input_data = input_data.replace(char, '')
        elif isinstance(input_data, dict):
            return {k: SecurityValidator.sanitize_input(v) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [SecurityValidator.sanitize_input(item) for item in input_data]
        
        return input_data

# Security validator instance
security_validator = SecurityValidator()

def validate_input_security(f):
    """Input validation decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check request data for security threats
        if request.is_json:
            data = request.get_json()
            if data:
                for key, value in data.items():
                    if isinstance(value, str):
                        if security_validator.detect_sql_injection(value):
                            logging.warning(f"Potential SQL injection detected from {request.remote_addr}: {value}")
                            return jsonify({
                                'error': 'Invalid input detected',
                                'message': 'Request contains potentially malicious content'
                            }), 400
        
        return f(*args, **kwargs)
    
    return decorated_function

# Rate limiting configurations for different endpoint types
def api_rate_limit(f):
    """Standard API rate limit"""
    return apply_rate_limit('api_key')(f)

def ml_rate_limit(f):
    """ML prediction rate limit"""
    return apply_rate_limit('ml_prediction')(f)

def ip_rate_limit(f):
    """IP-based rate limit"""
    return apply_rate_limit('ip')(f)

def endpoint_rate_limit(f):
    """Endpoint-specific rate limit"""
    return apply_rate_limit('endpoint')(f)