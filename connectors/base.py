import requests
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import time

class BaseConnector(ABC):
    """Base class for CRM connectors"""
    
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or self.get_api_key()
        self.base_url = base_url or self.get_base_url()
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # Default rate limiting
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'AI-Prediction-Platform/1.0',
            'Content-Type': 'application/json'
        })
        
        if self.api_key:
            self.session.headers.update(self.get_auth_headers())
    
    @abstractmethod
    def get_api_key(self):
        """Get API key from environment variables"""
        pass
    
    @abstractmethod
    def get_base_url(self):
        """Get base URL for the API"""
        pass
    
    @abstractmethod
    def get_auth_headers(self):
        """Get authentication headers"""
        pass
    
    def make_request(self, method, endpoint, params=None, data=None, timeout=30):
        """
        Make HTTP request with error handling and rate limiting
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict): Query parameters
            data (dict): Request body data
            timeout (int): Request timeout in seconds
            
        Returns:
            dict: API response
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=timeout
            )
            
            # Log request for debugging
            logging.debug(f"{method} {url} - Status: {response.status_code}")
            
            if response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                logging.warning(f"Rate limited, waiting {retry_after} seconds")
                time.sleep(retry_after)
                return self.make_request(method, endpoint, params, data, timeout)
            
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                return response.json()
            except ValueError:
                return {'raw_response': response.text}
                
        except requests.exceptions.Timeout:
            logging.error(f"Request timeout for {url}")
            raise Exception(f"Request timeout for {endpoint}")
        
        except requests.exceptions.ConnectionError:
            logging.error(f"Connection error for {url}")
            raise Exception(f"Connection error for {endpoint}")
        
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error for {url}: {e}")
            if response.status_code == 401:
                raise Exception("Authentication failed - check API key")
            elif response.status_code == 403:
                raise Exception("Access forbidden - insufficient permissions")
            elif response.status_code == 404:
                raise Exception(f"Endpoint not found: {endpoint}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            logging.error(f"Unexpected error for {url}: {str(e)}")
            raise Exception(f"Request failed: {str(e)}")
    
    def get(self, endpoint, params=None):
        """Make GET request"""
        return self.make_request('GET', endpoint, params=params)
    
    def post(self, endpoint, data=None, params=None):
        """Make POST request"""
        return self.make_request('POST', endpoint, params=params, data=data)
    
    def put(self, endpoint, data=None, params=None):
        """Make PUT request"""
        return self.make_request('PUT', endpoint, params=params, data=data)
    
    def delete(self, endpoint, params=None):
        """Make DELETE request"""
        return self.make_request('DELETE', endpoint, params=params)
    
    def test_connection(self):
        """Test API connection"""
        try:
            result = self.get_test_endpoint()
            return {
                'status': 'success',
                'message': 'Connection successful',
                'response': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    @abstractmethod
    def get_test_endpoint(self):
        """Get test endpoint for connection validation"""
        pass
    
    @abstractmethod
    def get_leads(self, limit=100, offset=0):
        """Get leads from CRM"""
        pass
    
    @abstractmethod
    def get_contacts(self, limit=100, offset=0):
        """Get contacts from CRM"""
        pass
    
    def normalize_lead_data(self, raw_data):
        """
        Normalize lead data to standard format
        
        Args:
            raw_data (dict): Raw lead data from CRM
            
        Returns:
            dict: Normalized lead data
        """
        # Default normalization - override in subclasses
        return {
            'id': raw_data.get('id'),
            'name': raw_data.get('name', ''),
            'email': raw_data.get('email', ''),
            'phone': raw_data.get('phone', ''),
            'company': raw_data.get('company', ''),
            'title': raw_data.get('title', ''),
            'source': raw_data.get('source', ''),
            'status': raw_data.get('status', ''),
            'created_date': raw_data.get('created_date'),
            'last_activity': raw_data.get('last_activity'),
            'score': raw_data.get('score', 0),
            'raw_data': raw_data
        }
    
    def normalize_contact_data(self, raw_data):
        """
        Normalize contact data to standard format
        
        Args:
            raw_data (dict): Raw contact data from CRM
            
        Returns:
            dict: Normalized contact data
        """
        # Default normalization - override in subclasses
        return {
            'id': raw_data.get('id'),
            'name': raw_data.get('name', ''),
            'email': raw_data.get('email', ''),
            'phone': raw_data.get('phone', ''),
            'company': raw_data.get('company', ''),
            'title': raw_data.get('title', ''),
            'lifecycle_stage': raw_data.get('lifecycle_stage', ''),
            'created_date': raw_data.get('created_date'),
            'last_activity': raw_data.get('last_activity'),
            'raw_data': raw_data
        }
    
    def get_rate_limit_info(self):
        """Get rate limit information"""
        return {
            'delay': self.rate_limit_delay,
            'headers': dict(self.session.headers)
        }
    
    def set_rate_limit(self, delay):
        """Set rate limit delay"""
        self.rate_limit_delay = max(0, delay)
    
    def get_supported_features(self):
        """Get list of supported features"""
        return {
            'leads': True,
            'contacts': True,
            'test_connection': True,
            'rate_limiting': True,
            'authentication': bool(self.api_key)
        }
