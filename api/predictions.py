from flask import Blueprint, request, jsonify
import time
import json
import numpy as np
from app import db, auth_required
from models import PredictionLog
from ml_services.data_cleaner import clean_api_data
from api.credit_manager import credit_required

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization and database storage"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def safe_log_prediction(user_id, model_type, input_data, prediction_result=None, confidence=None, processing_time=0.0, status='success', error_message=None):
    """Safely log prediction to database with proper type conversion"""
    try:
        # Convert all values to safe types
        if prediction_result:
            prediction_result = convert_numpy_types(prediction_result)
        
        # Ensure confidence is a proper Python float, not numpy type
        safe_confidence = None
        if confidence is not None:
            safe_confidence = float(convert_numpy_types(confidence))
        
        # Ensure processing_time is a proper Python float
        safe_processing_time = float(convert_numpy_types(processing_time))
        
        log = PredictionLog(
            user_id=int(user_id),
            model_type=str(model_type),
            input_data=json.dumps(convert_numpy_types(input_data)),
            prediction=json.dumps(prediction_result) if prediction_result else None,
            confidence=safe_confidence,
            processing_time=safe_processing_time,
            status=str(status),
            error_message=str(error_message) if error_message else None
        )
        db.session.add(log)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Database logging error: {e}")
        return False

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/lead-score', methods=['POST'])
@auth_required
@credit_required(credits_needed=1, operation_type='single_prediction')
def predict_lead_score(user_id):
    """Predict lead scoring"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Clean input data before prediction
        cleaned_data = clean_api_data(data, 'lead_score')
        
        from ml_services.lead_scoring import predict_lead_score as predict
        result = predict(cleaned_data)
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='lead_score',
            input_data=data,
            prediction_result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'lead_scoring_v1'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='lead_score',
            input_data=locals().get('data', {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500

@predictions_bp.route('/churn', methods=['POST'])
@auth_required
@credit_required(credits_needed=1, operation_type='single_prediction')
def predict_churn(user_id):
    """Predict customer churn"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Clean input data before prediction
        cleaned_data = clean_api_data(data, 'churn')
        
        from ml_services.churn_prediction import predict_churn as predict
        result = predict(cleaned_data)
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='churn',
            input_data=data,
            prediction_result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'churn_prediction_v1'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='churn',
            input_data=locals().get('data', {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500

@predictions_bp.route('/sales-forecast', methods=['POST'])
@auth_required
@credit_required(credits_needed=1, operation_type='single_prediction')
def predict_sales_forecast(user_id):
    """Predict sales forecast"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Clean input data before prediction
        cleaned_data = clean_api_data(data, 'sales_forecast')
        
        from ml_services.sales_forecast import predict_sales as predict
        result = predict(cleaned_data)
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='sales_forecast',
            input_data=data,
            prediction_result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'sales_forecast_v1'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='sales_forecast',
            input_data=locals().get('data', {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500

@predictions_bp.route('/nlp', methods=['POST'])
@auth_required
@credit_required(credits_needed=1, operation_type='single_prediction')
def predict_sentiment(user_id):
    """Analyze text sentiment"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field required'}), 400
        
        # Clean input data before prediction
        cleaned_data = clean_api_data(data, 'sentiment')
        
        from ml_services.nlp_service import analyze_sentiment
        result = analyze_sentiment(cleaned_data.get('text', data.get('text', '')))
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='sentiment',
            input_data=data,
            prediction_result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'sentiment_analysis_v1'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='sentiment',
            input_data=locals().get('data', {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500

@predictions_bp.route('/nlp/keywords', methods=['POST'])
@auth_required
@credit_required(credits_needed=1, operation_type='single_prediction')
def extract_keywords(user_id):
    """Extract keywords from text"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field required'}), 400
        
        # Clean input data before prediction
        cleaned_data = clean_api_data(data, 'keywords')
        
        from ml_services.nlp_service import extract_keywords as extract
        result = extract(cleaned_data.get('text', data.get('text', '')))
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='keywords',
            input_data=data,
            prediction_result=result,
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'keyword_extraction_v1'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='keywords',
            input_data=locals().get('data', {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500

@predictions_bp.route('/logs', methods=['GET'])
@auth_required
def get_prediction_logs(user_id):
    """Get prediction logs for the user"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        model_type = request.args.get('model_type')
        
        query = PredictionLog.query.filter_by(user_id=user_id)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        logs = query.order_by(PredictionLog.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'logs': [{
                'id': log.id,
                'model_type': log.model_type,
                'status': log.status,
                'confidence': log.confidence,
                'processing_time': log.processing_time,
                'created_at': log.created_at.isoformat(),
                'error_message': log.error_message
            } for log in logs.items],
            'pagination': {
                'page': logs.page,
                'per_page': logs.per_page,
                'total': logs.total,
                'pages': logs.pages
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
