"""
Specialized Industry-Specific ML Model Endpoints
Comprehensive API coverage for all industry models
"""
import json
import time
from flask import Blueprint, request, jsonify
from app import api_key_required
from api.credit_manager import credit_required
from models import PredictionLog, db
from api.predictions import safe_log_prediction

specialized_bp = Blueprint('specialized', __name__)

# ===== SPECIALIZED INDUSTRY MODEL ENDPOINTS =====

# Healthcare Industry Endpoints
@specialized_bp.route('/healthcare/risk-assessment', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='healthcare_model')
def healthcare_risk_assessment(user_id):
    """Healthcare-specific risk assessment"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Import and use healthcare model
        from ml_services.healthcare_models import predict_healthcare_risk
        result = predict_healthcare_risk(data)
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='healthcare_risk_assessment',
            input_data=data,
            prediction_result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'healthcare_risk_v1',
            'industry': 'healthcare'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='healthcare_risk_assessment',
            input_data=data if 'data' in locals() else {},
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500


# Finance Industry Endpoints
@specialized_bp.route('/finance/fraud-detection', methods=['POST'])
@api_key_required
@credit_required(credits_needed=5, operation_type='finance_model')
def finance_fraud_detection(user_id):
    """Financial fraud detection"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.finance_models import predict_fraud_detection
        result = predict_fraud_detection(data)
        
        processing_time = time.time() - start_time
        
        # Log the prediction safely
        safe_log_prediction(
            user_id=user_id,
            model_type='finance_fraud_detection',
            input_data=data,
            prediction_result=result,
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'finance_fraud_v1',
            'industry': 'finance'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error safely
        safe_log_prediction(
            user_id=user_id,
            model_type='finance_fraud_detection',
            input_data=data if 'data' in locals() else {},
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/finance/credit-scoring', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='finance_model')
def finance_credit_scoring(user_id):
    """Financial credit scoring"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.finance_models import predict_credit_score
        result = predict_credit_score(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='finance_credit_scoring',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'finance_credit_v1',
            'industry': 'finance'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='finance_credit_scoring',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


# Retail Industry Endpoints
@specialized_bp.route('/retail/price-optimization', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='retail_model')
def retail_price_optimization(user_id):
    """Retail price optimization"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.retail_models import predict_retail_price
        result = predict_retail_price(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='retail_price_optimization',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'retail_pricing_v1',
            'industry': 'retail'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='retail_price_optimization',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/retail/demand-forecast', methods=['POST'])
@api_key_required
@credit_required(credits_needed=3, operation_type='retail_model')
def retail_demand_forecast(user_id):
    """Retail demand forecasting"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.retail_models import predict_retail_demand
        result = predict_retail_demand(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='retail_demand_forecast',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'retail_demand_v1',
            'industry': 'retail'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='retail_demand_forecast',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


# SaaS Industry Endpoints
@specialized_bp.route('/saas/usage-prediction', methods=['POST'])
@api_key_required
@credit_required(credits_needed=3, operation_type='saas_model')
def saas_usage_prediction(user_id):
    """SaaS usage prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.saas_models import predict_saas_usage
        result = predict_saas_usage(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='saas_usage_prediction',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'saas_usage_v1',
            'industry': 'saas'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='saas_usage_prediction',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/saas/upsell-prediction', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='saas_model')
def saas_upsell_prediction(user_id):
    """SaaS upsell prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.saas_models import predict_saas_upsell
        result = predict_saas_upsell(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='saas_upsell_prediction',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'saas_upsell_v1',
            'industry': 'saas'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='saas_upsell_prediction',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


# Manufacturing Industry Endpoints
@specialized_bp.route('/manufacturing/quality-prediction', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='manufacturing_model')
def manufacturing_quality_prediction(user_id):
    """Manufacturing quality prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.manufacturing_models import predict_manufacturing_quality
        result = predict_manufacturing_quality(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='manufacturing_quality_prediction',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'manufacturing_quality_v1',
            'industry': 'manufacturing'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='manufacturing_quality_prediction',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/manufacturing/maintenance', methods=['POST'])
@api_key_required
@credit_required(credits_needed=5, operation_type='manufacturing_model')
def manufacturing_maintenance_prediction(user_id):
    """Manufacturing predictive maintenance"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.manufacturing_models import predict_maintenance_needs
        result = predict_maintenance_needs(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='manufacturing_maintenance',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'manufacturing_maintenance_v1',
            'industry': 'manufacturing'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='manufacturing_maintenance',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


# Education Industry Endpoints
@specialized_bp.route('/education/student-retention', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='education_model')
def education_student_retention(user_id):
    """Education student retention prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.education_models import predict_student_retention
        result = predict_student_retention(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='education_student_retention',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'education_retention_v1',
            'industry': 'education'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='education_student_retention',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/education/performance-prediction', methods=['POST'])
@api_key_required
@credit_required(credits_needed=3, operation_type='education_model')
def education_performance_prediction(user_id):
    """Education performance prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.education_models import predict_student_performance
        result = predict_student_performance(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='education_performance_prediction',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'education_performance_v1',
            'industry': 'education'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='education_performance_prediction',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


# Insurance Industry Endpoints
@specialized_bp.route('/insurance/risk-assessment', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='insurance_model')
def insurance_risk_assessment(user_id):
    """Insurance risk assessment"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.insurance_models import predict_insurance_risk
        result = predict_insurance_risk(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='insurance_risk_assessment',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'insurance_risk_v1',
            'industry': 'insurance'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='insurance_risk_assessment',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/insurance/claim-prediction', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='insurance_model')
def insurance_claim_prediction(user_id):
    """Insurance claim prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.insurance_models import predict_insurance_claims
        result = predict_insurance_claims(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='insurance_claim_prediction',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'insurance_claims_v1',
            'industry': 'insurance'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='insurance_claim_prediction',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


# Real Estate Industry Endpoints
@specialized_bp.route('/real-estate/price-prediction', methods=['POST'])
@api_key_required
@credit_required(credits_needed=4, operation_type='real_estate_model')
def real_estate_price_prediction(user_id):
    """Real estate price prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.real_estate_models import predict_real_estate_price
        result = predict_real_estate_price(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='real_estate_price_prediction',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'real_estate_price_v1',
            'industry': 'real_estate'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='real_estate_price_prediction',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@specialized_bp.route('/real-estate/market-analysis', methods=['POST'])
@api_key_required
@credit_required(credits_needed=3, operation_type='real_estate_model')
def real_estate_market_analysis(user_id):
    """Real estate market analysis"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        from ml_services.real_estate_models import predict_real_estate_market
        result = predict_real_estate_market(data)
        
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='real_estate_market_analysis',
            input_data=json.dumps(data),
            prediction=json.dumps(result),
            confidence=result.get('confidence', 0.0),
            processing_time=processing_time,
            status='success'
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'prediction': result,
            'processing_time': processing_time,
            'model': 'real_estate_market_v1',
            'industry': 'real_estate'
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        log = PredictionLog(
            user_id=user_id,
            model_type='real_estate_market_analysis',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500