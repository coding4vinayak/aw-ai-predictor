"""
Credit Management System

This module handles credit tracking, usage logging, and plan management
for the AI prediction platform.
"""

import json
import sys
from datetime import datetime, timedelta
from flask import request
from app import db
from models import UserCredit, CreditPlan, UsageLog, User, ApiKey


class CreditManager:
    """Manages user credits, usage tracking, and plan enforcement"""
    
    # Credit costs for different operations
    CREDIT_COSTS = {
        'single_prediction': 1,
        'batch_upload': 5,
        'batch_processing_per_row': 1,
        'large_file_extra': 2,  # For files > 5MB
        'industry_model': 3,    # Industry-specific models cost more
    }
    
    @staticmethod
    def get_user_credits(user_id: int, month_year: str = None) -> dict:
        """Get user's current credit status"""
        if not month_year:
            month_year = datetime.now().strftime('%Y-%m')
        
        credit_record = UserCredit.query.filter_by(
            user_id=user_id,
            month_year=month_year
        ).first()
        
        if not credit_record:
            # Create new credit record for the month with free plan
            free_plan = CreditPlan.query.filter_by(name='Free').first()
            if not free_plan:
                free_plan = CreditManager.create_default_plans()['free']
            
            credit_record = UserCredit(
                user_id=user_id,
                plan_id=free_plan.id,
                credits_remaining=free_plan.credits_per_month,
                month_year=month_year
            )
            db.session.add(credit_record)
            db.session.commit()
        
        return {
            'credits_used': credit_record.credits_used,
            'credits_remaining': credit_record.credits_remaining,
            'plan_name': credit_record.plan.name,
            'plan_limit': credit_record.plan.credits_per_month,
            'month_year': credit_record.month_year
        }
    
    @staticmethod
    def has_sufficient_credits(user_id: int, credits_needed: int) -> tuple:
        """Check if user has sufficient credits for operation"""
        credit_status = CreditManager.get_user_credits(user_id)
        
        has_credits = credit_status['credits_remaining'] >= credits_needed
        return has_credits, credit_status
    
    @staticmethod
    def deduct_credits(user_id: int, credits_to_deduct: int, 
                      operation_type: str, endpoint: str, 
                      data_size_kb: float = 0.0, processing_time: float = 0.0) -> bool:
        """Deduct credits from user account and log usage"""
        month_year = datetime.now().strftime('%Y-%m')
        
        credit_record = UserCredit.query.filter_by(
            user_id=user_id,
            month_year=month_year
        ).first()
        
        if not credit_record or credit_record.credits_remaining < credits_to_deduct:
            return False
        
        # Deduct credits
        credit_record.credits_used += credits_to_deduct
        credit_record.credits_remaining -= credits_to_deduct
        credit_record.updated_at = datetime.utcnow()
        
        # Log usage
        usage_log = UsageLog(
            user_id=user_id,
            api_key_id=getattr(request, 'api_key_id', None),
            endpoint=endpoint,
            method=request.method,
            credits_used=credits_to_deduct,
            data_size_kb=data_size_kb,
            processing_time=processing_time,
            status_code=200,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')[:500]
        )
        
        db.session.add(usage_log)
        db.session.commit()
        
        return True
    
    @staticmethod
    def calculate_credits_needed(operation_type: str, data_size_kb: float = 0.0, 
                               row_count: int = 1, is_industry_model: bool = False) -> int:
        """Calculate credits needed for an operation"""
        base_cost = CreditManager.CREDIT_COSTS.get(operation_type, 1)
        
        total_credits = base_cost
        
        # Add extra cost for large files (> 5MB)
        if data_size_kb > 5000:
            total_credits += CreditManager.CREDIT_COSTS['large_file_extra']
        
        # Add cost for batch processing
        if operation_type == 'batch_processing_per_row':
            total_credits = row_count * base_cost
        
        # Add cost for industry-specific models
        if is_industry_model:
            total_credits += CreditManager.CREDIT_COSTS['industry_model']
        
        return total_credits
    
    @staticmethod
    def create_default_plans() -> dict:
        """Create default credit plans"""
        plans_data = [
            {
                'name': 'Free',
                'credits_per_month': 100,
                'max_file_size_mb': 5,
                'max_requests_per_hour': 50,
                'price_per_month': 0.0,
                'features': json.dumps([
                    'Basic prediction models',
                    'Up to 100 credits/month',
                    'Standard support',
                    'Basic documentation'
                ])
            },
            {
                'name': 'Basic',
                'credits_per_month': 500,
                'max_file_size_mb': 16,
                'max_requests_per_hour': 200,
                'price_per_month': 29.99,
                'features': json.dumps([
                    'All prediction models',
                    'Up to 500 credits/month',
                    'Priority support',
                    'API documentation',
                    'Batch processing'
                ])
            },
            {
                'name': 'Premium',
                'credits_per_month': 2000,
                'max_file_size_mb': 50,
                'max_requests_per_hour': 500,
                'price_per_month': 99.99,
                'features': json.dumps([
                    'All prediction models',
                    'Industry-specific models',
                    'Up to 2000 credits/month',
                    'Priority support',
                    'Custom integrations',
                    'Advanced analytics'
                ])
            },
            {
                'name': 'Enterprise',
                'credits_per_month': 10000,
                'max_file_size_mb': 200,
                'max_requests_per_hour': 2000,
                'price_per_month': 499.99,
                'features': json.dumps([
                    'All features',
                    'Custom model training',
                    'Unlimited file size',
                    'Dedicated support',
                    'SLA guarantee',
                    'White-label options'
                ])
            }
        ]
        
        created_plans = {}
        for plan_data in plans_data:
            existing_plan = CreditPlan.query.filter_by(name=plan_data['name']).first()
            if not existing_plan:
                plan = CreditPlan(**plan_data)
                db.session.add(plan)
                created_plans[plan_data['name'].lower()] = plan
            else:
                created_plans[plan_data['name'].lower()] = existing_plan
        
        db.session.commit()
        return created_plans
    
    @staticmethod
    def get_usage_analytics(user_id: int = None, days: int = 30) -> dict:
        """Get usage analytics for admin or specific user"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        query = UsageLog.query.filter(UsageLog.timestamp >= start_date)
        if user_id:
            query = query.filter(UsageLog.user_id == user_id)
        
        logs = query.all()
        
        # Aggregate data
        analytics = {
            'total_requests': len(logs),
            'total_credits_used': sum(log.credits_used for log in logs),
            'unique_users': len(set(log.user_id for log in logs)),
            'avg_processing_time': sum(log.processing_time or 0 for log in logs) / len(logs) if logs else 0,
            'top_endpoints': {},
            'daily_usage': {},
            'error_rate': 0
        }
        
        # Endpoint usage
        endpoint_counts = {}
        for log in logs:
            endpoint_counts[log.endpoint] = endpoint_counts.get(log.endpoint, 0) + 1
        analytics['top_endpoints'] = dict(sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Daily usage
        for log in logs:
            day = log.timestamp.strftime('%Y-%m-%d')
            if day not in analytics['daily_usage']:
                analytics['daily_usage'][day] = {'requests': 0, 'credits': 0}
            analytics['daily_usage'][day]['requests'] += 1
            analytics['daily_usage'][day]['credits'] += log.credits_used
        
        # Error rate
        error_count = sum(1 for log in logs if log.status_code >= 400)
        analytics['error_rate'] = (error_count / len(logs) * 100) if logs else 0
        
        return analytics
    
    @staticmethod
    def upgrade_user_plan(user_id: int, new_plan_name: str) -> bool:
        """Upgrade user to new plan"""
        new_plan = CreditPlan.query.filter_by(name=new_plan_name, is_active=True).first()
        if not new_plan:
            return False
        
        month_year = datetime.now().strftime('%Y-%m')
        credit_record = UserCredit.query.filter_by(
            user_id=user_id,
            month_year=month_year
        ).first()
        
        if credit_record:
            # Update existing record
            credit_record.plan_id = new_plan.id
            credit_record.credits_remaining = new_plan.credits_per_month - credit_record.credits_used
            credit_record.updated_at = datetime.utcnow()
        else:
            # Create new record
            credit_record = UserCredit(
                user_id=user_id,
                plan_id=new_plan.id,
                credits_remaining=new_plan.credits_per_month,
                month_year=month_year
            )
            db.session.add(credit_record)
        
        db.session.commit()
        return True


def credit_required(credits_needed: int = 1, operation_type: str = 'single_prediction'):
    """Decorator to check and deduct credits for API endpoints"""
    def decorator(f):
        def decorated_function(*args, **kwargs):
            # Extract user_id from function arguments
            user_id = kwargs.get('user_id') or (args[0] if args else None)
            
            if not user_id:
                return {'error': 'User ID required'}, 401
            
            # Calculate data size
            data_size_kb = 0.0
            if request.json:
                data_size_kb = sys.getsizeof(json.dumps(request.json)) / 1024
            elif request.files:
                for file in request.files.values():
                    file.seek(0, 2)  # Seek to end
                    data_size_kb += file.tell() / 1024
                    file.seek(0)  # Reset file pointer
            
            # Check if industry model is being used
            is_industry_model = 'industry' in request.path
            
            # Calculate actual credits needed
            actual_credits = CreditManager.calculate_credits_needed(
                operation_type, data_size_kb, is_industry_model=is_industry_model
            )
            
            # Check credits
            has_credits, credit_status = CreditManager.has_sufficient_credits(user_id, actual_credits)
            
            if not has_credits:
                return {
                    'error': 'Insufficient credits',
                    'credits_needed': actual_credits,
                    'credits_remaining': credit_status['credits_remaining'],
                    'plan_name': credit_status['plan_name']
                }, 402  # Payment Required
            
            # Execute the function
            start_time = datetime.now()
            try:
                result = f(*args, **kwargs)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Deduct credits on successful operation
                CreditManager.deduct_credits(
                    user_id, actual_credits, operation_type,
                    request.path, data_size_kb, processing_time
                )
                
                # Add credit info to response if it's a dict
                if isinstance(result, tuple) and len(result) == 2:
                    response_data, status_code = result
                    if isinstance(response_data, dict):
                        response_data['credits_used'] = actual_credits
                        response_data['credits_remaining'] = credit_status['credits_remaining'] - actual_credits
                        return response_data, status_code
                elif isinstance(result, dict):
                    result['credits_used'] = actual_credits
                    result['credits_remaining'] = credit_status['credits_remaining'] - actual_credits
                
                return result
                
            except Exception as e:
                # Log error usage without deducting credits
                usage_log = UsageLog(
                    user_id=user_id,
                    endpoint=request.path,
                    method=request.method,
                    credits_used=0,
                    data_size_kb=data_size_kb,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    status_code=500,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', '')[:500]
                )
                db.session.add(usage_log)
                db.session.commit()
                raise e
        
        decorated_function.__name__ = f.__name__
        return decorated_function
    return decorator