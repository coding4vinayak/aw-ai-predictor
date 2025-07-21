import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from .utils import preprocess_features, create_dummy_model
from monitoring.metrics import track_prediction_performance
from performance.caching import cache_prediction

def get_model():
    """Load or create lead scoring model"""
    try:
        from .model_manager import model_manager
        return model_manager.get_active_model('lead_scoring')
    except ImportError:
        # Fallback to original logic if model_manager not available
        model_path = "ml_models/lead_scoring_model.pkl"
        scaler_path = "ml_models/lead_scoring_scaler.pkl"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                logging.info("Loaded existing lead scoring model")
            else:
                # Create dummy model if none exists
                model, scaler = create_dummy_model('lead_scoring')
                logging.info("Created dummy lead scoring model")
            
            return model, scaler
        except Exception as e:
            logging.error(f"Error loading lead scoring model: {str(e)}")
            # Fallback to creating a new dummy model
            model, scaler = create_dummy_model('lead_scoring')
            return model, scaler

@cache_prediction('lead_scoring', ttl=1800)
@track_prediction_performance('lead_scoring')
def predict_lead_score(data):
    """
    Predict lead score based on input data
    
    Args:
        data (dict): Input features for lead scoring
        
    Returns:
        dict: Prediction results with score and confidence
    """
    try:
        model, scaler = get_model()
        
        # Expected features for lead scoring
        expected_features = [
            'company_size', 'budget', 'industry_score', 'engagement_score',
            'demographic_score', 'behavioral_score', 'source_score'
        ]
        
        # Preprocess input data
        features = preprocess_features(data, expected_features)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get prediction and probability
        prediction = model.predict(features_scaled)[0]
        
        # Handle both classifiers and regressors
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            # For regressors, use normalized prediction as confidence
            confidence = min(1.0, max(0.0, prediction / 100.0)) if prediction >= 0 else 0.5
        
        # Calculate score (0-100)
        score = confidence * 100
        
        # Determine quality level
        if score >= 80:
            quality = "Hot"
        elif score >= 60:
            quality = "Warm"
        elif score >= 40:
            quality = "Cold"
        else:
            quality = "Unqualified"
        
        return {
            'score': round(score, 2),
            'confidence': round(confidence, 4),
            'quality': quality,
            'prediction': int(prediction),
            'features_used': expected_features,
            'model_version': 'v1.0'
        }
        
    except Exception as e:
        logging.error(f"Lead scoring prediction error: {str(e)}")
        # Return default/fallback prediction
        return {
            'score': 50.0,
            'confidence': 0.5,
            'quality': 'Unknown',
            'prediction': 0,
            'error': str(e),
            'model_version': 'v1.0'
        }

def train_model(training_data):
    """
    Train a new lead scoring model with provided data
    
    Args:
        training_data (pd.DataFrame): Training dataset
        
    Returns:
        dict: Training results
    """
    try:
        # Prepare features and target
        feature_columns = [
            'company_size', 'budget', 'industry_score', 'engagement_score',
            'demographic_score', 'behavioral_score', 'source_score'
        ]
        
        X = training_data[feature_columns]
        y = training_data['converted']  # Assuming this is the target column
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Save model and scaler
        os.makedirs('ml_models', exist_ok=True)
        joblib.dump(model, 'ml_models/lead_scoring_model.pkl')
        joblib.dump(scaler, 'ml_models/lead_scoring_scaler.pkl')
        
        # Calculate training score
        train_score = model.score(X_scaled, y)
        
        return {
            'message': 'Model trained successfully',
            'train_accuracy': round(train_score, 4),
            'features': feature_columns,
            'model_type': 'RandomForestClassifier'
        }
        
    except Exception as e:
        logging.error(f"Lead scoring model training error: {str(e)}")
        return {
            'error': str(e),
            'message': 'Model training failed'
        }

def get_feature_importance():
    """Get feature importance from the trained model"""
    try:
        model, _ = get_model()
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'company_size', 'budget', 'industry_score', 'engagement_score',
                'demographic_score', 'behavioral_score', 'source_score'
            ]
            
            importances = model.feature_importances_
            
            feature_importance = [
                {'feature': name, 'importance': round(imp, 4)}
                for name, imp in zip(feature_names, importances)
            ]
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'feature_importance': feature_importance,
                'model_type': type(model).__name__
            }
        else:
            return {
                'error': 'Model does not support feature importance',
                'model_type': type(model).__name__
            }
            
    except Exception as e:
        logging.error(f"Feature importance error: {str(e)}")
        return {
            'error': str(e)
        }
