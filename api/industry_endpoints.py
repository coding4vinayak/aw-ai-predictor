"""
Industry-Specific API Endpoints

This module provides industry-specific prediction endpoints with
specialized models and preprocessing.

Endpoints follow pattern: /api/{industry}/{model_type}
Example: /api/healthcare/churn, /api/finance/lead-score
"""

from flask import Blueprint, request, jsonify
import time
import json
from app import db, api_key_required
from models import PredictionLog, IndustryModel
from api.credit_manager import credit_required
from ml_services.data_cleaner import clean_api_data

industry_bp = Blueprint('industry', __name__)

# Supported industries and their specialized models
INDUSTRY_MODELS = {
    'healthcare': ['churn', 'lead-score', 'sentiment', 'risk-assessment', 'clinical-outcomes'],
    'finance': ['churn', 'lead-score', 'fraud-detection', 'credit-scoring', 'portfolio-risk'],
    'retail': ['churn', 'lead-score', 'demand-forecast', 'price-optimization', 'inventory-optimization'],
    'saas': ['churn', 'lead-score', 'usage-prediction', 'upsell-prediction', 'feature-adoption'],
    'real-estate': ['lead-score', 'price-prediction', 'market-analysis', 'investment-scoring'],
    'insurance': ['churn', 'claim-prediction', 'risk-assessment', 'fraud-detection', 'underwriting'],
    'manufacturing': ['demand-forecast', 'quality-prediction', 'maintenance', 'supply-chain'],
    'education': ['student-retention', 'performance-prediction', 'engagement', 'learning-analytics'],
}

def get_industry_model(industry: str, model_type: str):
    """Get industry-specific model configuration"""
    # Clean up model type names
    model_type_mapping = {
        'lead-score': 'lead_score',
        'churn': 'churn', 
        'sentiment': 'sentiment',
        'fraud-detection': 'fraud_detection',
        'credit-scoring': 'credit_scoring',
        'demand-forecast': 'demand_forecast',
        'price-optimization': 'price_optimization',
        'usage-prediction': 'usage_prediction',
        'upsell-prediction': 'upsell_prediction',
        'price-prediction': 'price_prediction',
        'market-analysis': 'market_analysis',
        'claim-prediction': 'claim_prediction',
        'risk-assessment': 'risk_assessment',
        'quality-prediction': 'quality_prediction',
        'maintenance': 'maintenance',
        'student-retention': 'student_retention',
        'performance-prediction': 'performance_prediction',
        'engagement': 'engagement',
        'clinical-outcomes': 'clinical_outcomes',
        'portfolio-risk': 'portfolio_risk',
        'inventory-optimization': 'inventory_optimization',
        'feature-adoption': 'feature_adoption',
        'investment-scoring': 'investment_scoring',
        'underwriting': 'underwriting',
        'supply-chain': 'supply_chain',
        'learning-analytics': 'learning_analytics'
    }
    
    normalized_model_type = model_type_mapping.get(model_type, model_type)
    
    # Look for industry-specific model
    industry_model = IndustryModel.query.filter_by(
        industry_name=industry,
        model_type=normalized_model_type,
        is_active=True
    ).first()
    
    return industry_model


@industry_bp.route('/<industry>/churn', methods=['POST'])
@api_key_required
@credit_required(credits_needed=3, operation_type='industry_model')
def predict_industry_churn(user_id, industry):
    """Industry-specific churn prediction"""
    start_time = time.time()
    
    try:
        # Validate industry
        if industry not in INDUSTRY_MODELS or 'churn' not in INDUSTRY_MODELS[industry]:
            return jsonify({'error': f'Churn prediction not available for {industry} industry'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Get industry-specific model configuration
        industry_model = get_industry_model(industry, 'churn')
        
        # Clean input data with industry-specific preprocessing
        cleaned_data = clean_api_data(data, 'churn')
        
        # Apply industry-specific preprocessing if available
        if industry_model and industry_model.preprocessing_config:
            preprocessing = json.loads(industry_model.preprocessing_config)
            cleaned_data = apply_industry_preprocessing(cleaned_data, preprocessing, industry)
        
        # Get prediction using industry-optimized logic
        from ml_services.churn_prediction import predict_churn as predict
        result = predict(cleaned_data)
        
        # Apply industry-specific adjustments
        result = apply_industry_adjustments(result, industry, 'churn')
        
        processing_time = time.time() - start_time
        
        # Log the prediction
        log = PredictionLog(
            user_id=user_id,
            model_type=f'{industry}_churn',
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
            'model': f'{industry}_churn_v1',
            'industry': industry,
            'model_accuracy': industry_model.accuracy_score if industry_model else None
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error
        log = PredictionLog(
            user_id=user_id,
            model_type=f'{industry}_churn',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@industry_bp.route('/<industry>/lead-score', methods=['POST'])
@api_key_required
@credit_required(credits_needed=3, operation_type='industry_model')
def predict_industry_lead_score(user_id, industry):
    """Industry-specific lead scoring"""
    start_time = time.time()
    
    try:
        # Validate industry
        if industry not in INDUSTRY_MODELS or 'lead-score' not in INDUSTRY_MODELS[industry]:
            return jsonify({'error': f'Lead scoring not available for {industry} industry'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Get industry-specific model configuration
        industry_model = get_industry_model(industry, 'lead-score')
        
        # Clean input data with industry-specific preprocessing
        cleaned_data = clean_api_data(data, 'lead_score')
        
        # Apply industry-specific preprocessing if available
        if industry_model and industry_model.preprocessing_config:
            preprocessing = json.loads(industry_model.preprocessing_config)
            cleaned_data = apply_industry_preprocessing(cleaned_data, preprocessing, industry)
        
        # Get prediction using industry-optimized logic
        from ml_services.lead_scoring import predict_lead_score as predict
        result = predict(cleaned_data)
        
        # Apply industry-specific adjustments
        result = apply_industry_adjustments(result, industry, 'lead_score')
        
        processing_time = time.time() - start_time
        
        # Log the prediction
        log = PredictionLog(
            user_id=user_id,
            model_type=f'{industry}_lead_score',
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
            'model': f'{industry}_lead_score_v1',
            'industry': industry,
            'model_accuracy': industry_model.accuracy_score if industry_model else None
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error
        log = PredictionLog(
            user_id=user_id,
            model_type=f'{industry}_lead_score',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@industry_bp.route('/<industry>/sentiment', methods=['POST'])
@api_key_required
@credit_required(credits_needed=2, operation_type='industry_model')
def predict_industry_sentiment(user_id, industry):
    """Industry-specific sentiment analysis"""
    start_time = time.time()
    
    try:
        # Validate industry
        if industry not in INDUSTRY_MODELS:
            return jsonify({'error': f'Industry {industry} not supported'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field required'}), 400
        
        # Clean input data with industry-specific preprocessing
        cleaned_data = clean_api_data(data, 'sentiment')
        
        # Get prediction with industry context
        from ml_services.nlp_service import analyze_sentiment
        result = analyze_sentiment(cleaned_data.get('text', data.get('text', '')))
        
        # Apply industry-specific sentiment adjustments
        result = apply_industry_sentiment_adjustments(result, industry)
        
        processing_time = time.time() - start_time
        
        # Log the prediction
        log = PredictionLog(
            user_id=user_id,
            model_type=f'{industry}_sentiment',
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
            'model': f'{industry}_sentiment_v1',
            'industry': industry
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error
        log = PredictionLog(
            user_id=user_id,
            model_type=f'{industry}_sentiment',
            input_data=json.dumps(data if 'data' in locals() else {}),
            processing_time=processing_time,
            status='error',
            error_message=str(e)
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'error': str(e)}), 500


@industry_bp.route('/industries', methods=['GET'])
def list_industries():
    """List supported industries and their available models"""
    return jsonify({
        'supported_industries': INDUSTRY_MODELS,
        'total_industries': len(INDUSTRY_MODELS),
        'endpoint_pattern': '/api/industry/{industry}/{model_type}'
    })


def apply_industry_preprocessing(data, preprocessing_config, industry):
    """Apply industry-specific preprocessing rules"""
    
    industry_rules = {
        'healthcare': {
            'age_groups': {'0-18': 'pediatric', '19-64': 'adult', '65+': 'senior'},
            'normalize_medical_terms': True
        },
        'finance': {
            'income_buckets': {'0-30000': 'low', '30000-80000': 'medium', '80000+': 'high'},
            'credit_score_normalization': True
        },
        'retail': {
            'seasonality_adjustment': True,
            'category_mapping': True
        },
        'saas': {
            'usage_metrics_normalization': True,
            'feature_adoption_scoring': True
        }
    }
    
    if industry not in industry_rules:
        return data
    
    rules = industry_rules[industry]
    processed_data = data.copy()
    
    # Apply industry-specific transformations
    if industry == 'healthcare' and rules.get('normalize_medical_terms'):
        # Normalize medical terminology
        if 'symptoms' in processed_data:
            processed_data['symptoms'] = normalize_medical_terms(processed_data['symptoms'])
    
    elif industry == 'finance' and rules.get('credit_score_normalization'):
        # Normalize credit scores to 0-1 range
        if 'credit_score' in processed_data:
            processed_data['credit_score_normalized'] = float(processed_data['credit_score']) / 850
    
    elif industry == 'retail' and rules.get('seasonality_adjustment'):
        # Add seasonal indicators
        from datetime import datetime
        month = datetime.now().month
        processed_data['season'] = get_season(month)
    
    return processed_data


def apply_industry_adjustments(result, industry, model_type):
    """Apply industry-specific adjustments to prediction results"""
    
    adjustments = {
        'healthcare': {
            'churn': {'risk_factors': ['health_outcomes', 'cost_concerns', 'provider_changes']},
            'lead_score': {'key_indicators': ['insurance_type', 'health_history', 'demographic_match']}
        },
        'finance': {
            'churn': {'risk_factors': ['fee_sensitivity', 'service_usage', 'competitor_offers']},
            'lead_score': {'key_indicators': ['credit_profile', 'income_stability', 'existing_relationship']}
        },
        'retail': {
            'churn': {'risk_factors': ['purchase_frequency', 'seasonal_patterns', 'price_sensitivity']},
            'lead_score': {'key_indicators': ['brand_affinity', 'purchase_history', 'channel_preference']}
        }
    }
    
    if industry in adjustments and model_type in adjustments[industry]:
        industry_context = adjustments[industry][model_type]
        result['industry_context'] = industry_context
        
        # Add industry-specific recommendations
        if model_type == 'churn' and 'risk_factors' in industry_context:
            result['recommended_actions'] = get_churn_recommendations(industry, result.get('prediction', 0))
        elif model_type == 'lead_score' and 'key_indicators' in industry_context:
            result['engagement_strategy'] = get_lead_engagement_strategy(industry, result.get('prediction', 0))
    
    return result


def apply_industry_sentiment_adjustments(result, industry):
    """Apply industry-specific sentiment analysis adjustments"""
    
    industry_contexts = {
        'healthcare': {
            'keywords': ['treatment', 'care', 'doctor', 'hospital', 'recovery'],
            'sentiment_weight': 1.2  # Healthcare sentiment is more critical
        },
        'finance': {
            'keywords': ['money', 'investment', 'loan', 'credit', 'payment'],
            'sentiment_weight': 1.1
        },
        'retail': {
            'keywords': ['product', 'service', 'price', 'quality', 'delivery'],
            'sentiment_weight': 1.0
        }
    }
    
    if industry in industry_contexts:
        context = industry_contexts[industry]
        
        # Check for industry-specific keywords
        text_lower = result.get('text', '').lower()
        keyword_matches = [kw for kw in context['keywords'] if kw in text_lower]
        
        if keyword_matches:
            result['industry_keywords'] = keyword_matches
            # Adjust confidence based on industry context
            original_confidence = result.get('confidence', 0.5)
            result['confidence'] = min(1.0, original_confidence * context['sentiment_weight'])
            result['industry_adjusted'] = True
    
    return result


def normalize_medical_terms(text):
    """Normalize medical terminology"""
    # Simple medical term normalization
    medical_mappings = {
        'hypertension': 'high blood pressure',
        'myocardial infarction': 'heart attack',
        'cerebrovascular accident': 'stroke',
        'diabetes mellitus': 'diabetes'
    }
    
    normalized_text = text.lower()
    for medical_term, common_term in medical_mappings.items():
        normalized_text = normalized_text.replace(medical_term, common_term)
    
    return normalized_text


def get_season(month):
    """Get season from month number"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def get_churn_recommendations(industry, churn_probability):
    """Get industry-specific churn prevention recommendations"""
    
    recommendations = {
        'healthcare': {
            'high': ['Improve care coordination', 'Reduce wait times', 'Enhance communication'],
            'medium': ['Follow up on treatments', 'Provide health education', 'Improve accessibility'],
            'low': ['Maintain relationship', 'Preventive care reminders', 'Wellness programs']
        },
        'finance': {
            'high': ['Review fees and rates', 'Offer retention incentives', 'Personal consultation'],
            'medium': ['Improve digital experience', 'Provide financial education', 'Cross-sell relevant products'],
            'low': ['Regular check-ins', 'Market rate reviews', 'Loyalty rewards']
        }
    }
    
    risk_level = 'high' if churn_probability > 0.7 else 'medium' if churn_probability > 0.4 else 'low'
    
    return recommendations.get(industry, {}).get(risk_level, ['Standard retention strategies'])


def get_lead_engagement_strategy(industry, lead_score):
    """Get industry-specific lead engagement strategies"""
    
    strategies = {
        'healthcare': {
            'high': ['Schedule consultation', 'Provide personalized care plan', 'Fast-track scheduling'],
            'medium': ['Educational content', 'Care coordinator outreach', 'Insurance verification'],
            'low': ['Newsletter signup', 'Health tips', 'Community events']
        },
        'finance': {
            'high': ['Personal advisor meeting', 'Customized financial plan', 'Priority processing'],
            'medium': ['Product demos', 'Educational webinars', 'Rate comparisons'],
            'low': ['Newsletter', 'Market updates', 'General information']
        }
    }
    

