"""
Admin Panel API

This module provides admin functionality for managing users,
credits, plans, and analytics.
"""

from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for
from datetime import datetime, timedelta
from app import db
from models import User, ApiKey, CreditPlan, UserCredit, UsageLog, IndustryModel
from api.credit_manager import CreditManager
import json

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """Decorator to require admin privileges"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if not user or user.username != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function


@admin_bp.route('/')
@admin_required
def admin_dashboard():
    """Admin dashboard with overview"""
    # Get system statistics
    stats = {
        'total_users': User.query.count(),
        'active_users': User.query.filter_by(is_active=True).count(),
        'total_api_keys': ApiKey.query.filter_by(is_active=True).count(),
        'total_plans': CreditPlan.query.filter_by(is_active=True).count(),
    }
    
    # Get usage analytics for last 30 days
    analytics = CreditManager.get_usage_analytics(days=30)
    
    # Get recent activity
    recent_logs = UsageLog.query.order_by(UsageLog.timestamp.desc()).limit(10).all()
    
    return render_template('admin/dashboard.html', 
                         stats=stats, 
                         analytics=analytics, 
                         recent_logs=recent_logs)


@admin_bp.route('/users')
@admin_required
def manage_users():
    """User management page"""
    page = request.args.get('page', 1, type=int)
    users = User.query.paginate(
        page=page, per_page=20, error_out=False
    )
    
    return render_template('admin/users.html', users=users)


@admin_bp.route('/users/<int:user_id>')
@admin_required
def user_details(user_id):
    """User details and credit management"""
    user = User.query.get_or_404(user_id)
    
    # Get user's credit history
    credits = UserCredit.query.filter_by(user_id=user_id).order_by(UserCredit.month_year.desc()).all()
    
    # Get user's usage logs
    usage_logs = UsageLog.query.filter_by(user_id=user_id).order_by(UsageLog.timestamp.desc()).limit(50).all()
    
    # Get current month credits
    current_month = datetime.now().strftime('%Y-%m')
    current_credits = CreditManager.get_user_credits(user_id, current_month)
    
    return render_template('admin/user_details.html', 
                         user=user, 
                         credits=credits, 
                         usage_logs=usage_logs,
                         current_credits=current_credits)


@admin_bp.route('/api/users/<int:user_id>/credits', methods=['POST'])
@admin_required
def update_user_credits():
    """Update user credits (admin only)"""
    data = request.get_json()
    user_id = data.get('user_id')
    action = data.get('action')  # add, set, upgrade_plan
    amount = data.get('amount', 0)
    plan_name = data.get('plan_name')
    
    if not user_id or not action:
        return jsonify({'error': 'User ID and action required'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    month_year = datetime.now().strftime('%Y-%m')
    credit_record = UserCredit.query.filter_by(
        user_id=user_id, month_year=month_year
    ).first()
    
    if not credit_record:
        # Create new credit record with free plan
        free_plan = CreditPlan.query.filter_by(name='Free').first()
        credit_record = UserCredit()
        credit_record.user_id = user_id
        credit_record.plan_id = free_plan.id if free_plan else 1
        credit_record.credits_remaining = free_plan.credits_per_month if free_plan else 100
        credit_record.month_year = month_year
        db.session.add(credit_record)
    
    if action == 'add':
        credit_record.credits_remaining += amount
    elif action == 'set':
        credit_record.credits_remaining = amount
    elif action == 'upgrade_plan':
        success = CreditManager.upgrade_user_plan(user_id, plan_name)
        if not success:
            return jsonify({'error': 'Failed to upgrade plan'}), 400
    
    credit_record.updated_at = datetime.utcnow()
    db.session.commit()
    
    # Get plan name safely
    plan_name = 'Unknown'
    if credit_record.plan_id:
        plan = CreditPlan.query.get(credit_record.plan_id)
        plan_name = plan.name if plan else 'Unknown'
    
    return jsonify({
        'success': True,
        'credits_remaining': credit_record.credits_remaining,
        'plan_name': plan_name
    })


@admin_bp.route('/plans')
@admin_required
def manage_plans():
    """Credit plan management"""
    plans = CreditPlan.query.all()
    return render_template('admin/plans.html', plans=plans)


@admin_bp.route('/api/plans', methods=['POST'])
@admin_required
def create_plan():
    """Create new credit plan"""
    data = request.get_json()
    
    required_fields = ['name', 'credits_per_month', 'price_per_month']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400
    
    plan = CreditPlan()
    plan.name = data['name']
    plan.credits_per_month = data['credits_per_month']
    plan.max_file_size_mb = data.get('max_file_size_mb', 16)
    plan.max_requests_per_hour = data.get('max_requests_per_hour', 100)
    plan.price_per_month = data['price_per_month']
    plan.features = json.dumps(data.get('features', []))
    plan.is_active = data.get('is_active', True)
    
    db.session.add(plan)
    db.session.commit()
    
    return jsonify({'success': True, 'plan_id': plan.id})


@admin_bp.route('/api/plans/<int:plan_id>', methods=['PUT'])
@admin_required
def update_plan(plan_id):
    """Update credit plan"""
    plan = CreditPlan.query.get_or_404(plan_id)
    data = request.get_json()
    
    # Update fields
    for field in ['name', 'credits_per_month', 'max_file_size_mb', 
                  'max_requests_per_hour', 'price_per_month', 'is_active']:
        if field in data:
            setattr(plan, field, data[field])
    
    if 'features' in data:
        plan.features = json.dumps(data['features'])
    
    db.session.commit()
    
    return jsonify({'success': True})


@admin_bp.route('/analytics')
@admin_required
def analytics_dashboard():
    """Analytics and usage statistics"""
    days = request.args.get('days', 30, type=int)
    analytics = CreditManager.get_usage_analytics(days=days)
    
    # Get plan distribution
    plan_stats = db.session.query(
        CreditPlan.name, 
        db.func.count(UserCredit.id)
    ).join(UserCredit).group_by(CreditPlan.name).all()
    
    analytics['plan_distribution'] = dict(plan_stats)
    
    return render_template('admin/analytics.html', analytics=analytics, days=days)


@admin_bp.route('/models')
@admin_required
def manage_models():
    """Industry model management"""
    models = IndustryModel.query.all()
    return render_template('admin/models.html', models=models)


@admin_bp.route('/api/models', methods=['POST'])
@admin_required
def create_industry_model():
    """Create new industry-specific model"""
    data = request.get_json()
    
    required_fields = ['industry_name', 'model_type']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400
    
    model = IndustryModel()
    model.industry_name = data['industry_name']
    model.model_type = data['model_type']
    model.model_version = data.get('model_version', 'v1.0')
    model.model_path = data.get('model_path')
    model.is_active = data.get('is_active', True)
    model.accuracy_score = data.get('accuracy_score')
    model.required_features = json.dumps(data.get('required_features', []))
    model.feature_weights = json.dumps(data.get('feature_weights', {}))
    model.preprocessing_config = json.dumps(data.get('preprocessing_config', {}))
    
    db.session.add(model)
    db.session.commit()
    
    return jsonify({'success': True, 'model_id': model.id})


@admin_bp.route('/api/models/<int:model_id>', methods=['PUT'])
@admin_required
def update_industry_model(model_id):
    """Update industry model"""
    model = IndustryModel.query.get_or_404(model_id)
    data = request.get_json()
    
    # Update basic fields
    for field in ['industry_name', 'model_type', 'model_version', 
                  'model_path', 'is_active', 'accuracy_score']:
        if field in data:
            setattr(model, field, data[field])
    
    # Update JSON fields
    json_fields = ['required_features', 'feature_weights', 'preprocessing_config']
    for field in json_fields:
        if field in data:
            setattr(model, field, json.dumps(data[field]))
    
    model.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True})


@admin_bp.route('/api/models/<int:model_id>', methods=['DELETE'])
@admin_required
def delete_industry_model(model_id):
    """Delete industry model"""
    model = IndustryModel.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    
    return jsonify({'success': True})


def setup_default_data():
    """Setup default credit plans and admin user"""
    # Create default plans
    CreditManager.create_default_plans()
    
    # Ensure admin user exists (but don't create if already exists)
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        try:
            from werkzeug.security import generate_password_hash
            admin_user = User()
            admin_user.username = 'admin'
            admin_user.email = 'admin@example.com'
            admin_user.password_hash = generate_password_hash('admin123')
            db.session.add(admin_user)
            db.session.commit()
        except Exception as e:
            # If user already exists, just continue
            db.session.rollback()
            admin_user = User.query.filter_by(username='admin').first()