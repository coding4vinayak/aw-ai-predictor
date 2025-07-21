"""
Real-time monitoring dashboard for AI platform
Enterprise-grade observability with model performance tracking
"""

from flask import Blueprint, render_template, jsonify, request
from datetime import datetime, timedelta
import json
from .metrics import model_monitor, metrics_store, get_system_health
from app import auth_required

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/monitoring')
@auth_required
def monitoring_dashboard(user_id):
    """Main monitoring dashboard"""
    return render_template('monitoring/dashboard.html')

@dashboard_bp.route('/monitoring/api/health')
@auth_required
def api_health(user_id):
    """System health endpoint"""
    return jsonify(get_system_health())

@dashboard_bp.route('/monitoring/api/models')
@auth_required
def api_models(user_id):
    """Model performance metrics"""
    return jsonify(model_monitor.get_dashboard_data())

@dashboard_bp.route('/monitoring/api/metrics')
@auth_required
def api_metrics(user_id):
    """Raw metrics data"""
    time_window = request.args.get('window', 60, type=int)
    return jsonify(metrics_store.get_metrics_summary(time_window))

@dashboard_bp.route('/monitoring/api/alerts')
@auth_required
def api_alerts(user_id):
    """Active alerts"""
    dashboard_data = model_monitor.get_dashboard_data()
    return jsonify({
        'alerts': dashboard_data.get('alerts', []),
        'count': len(dashboard_data.get('alerts', [])),
        'timestamp': datetime.utcnow().isoformat()
    })

@dashboard_bp.route('/monitoring/api/model/<model_name>')
@auth_required
def api_model_details(user_id, model_name):
    """Detailed metrics for specific model"""
    dashboard_data = model_monitor.get_dashboard_data()
    model_data = dashboard_data.get('models', {}).get(model_name, {})
    
    if not model_data:
        return jsonify({'error': 'Model not found'}), 404
    
    # Get additional time-series data
    time_window = request.args.get('window', 60, type=int)
    metrics_summary = metrics_store.get_metrics_summary(time_window)
    
    # Filter metrics for this model
    model_metrics = {}
    for metric_name, metric_data in metrics_summary.items():
        if model_name in metric_name:
            model_metrics[metric_name] = metric_data
    
    return jsonify({
        'model': model_data,
        'metrics': model_metrics,
        'timestamp': datetime.utcnow().isoformat()
    })

@dashboard_bp.route('/monitoring/api/usage-stats')
@auth_required
def api_usage_stats(user_id):
    """API usage statistics"""
    dashboard_data = model_monitor.get_dashboard_data()
    
    # Calculate usage statistics
    total_predictions = sum(
        model_data.get('predictions_count', 0) 
        for model_data in dashboard_data.get('models', {}).values()
    )
    
    avg_response_time = 0
    model_count = len(dashboard_data.get('models', {}))
    if model_count > 0:
        avg_response_time = sum(
            model_data.get('avg_processing_time', 0)
            for model_data in dashboard_data.get('models', {}).values()
        ) / model_count
    
    return jsonify({
        'total_predictions': total_predictions,
        'active_models': model_count,
        'avg_response_time': round(avg_response_time, 3),
        'system_uptime': '99.9%',  # Calculate from actual uptime in production
        'timestamp': datetime.utcnow().isoformat()
    })