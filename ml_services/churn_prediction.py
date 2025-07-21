import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from .utils import preprocess_features, create_dummy_model

def get_model():
    """Load or create churn prediction model"""
    model_path = "ml_models/churn_model.pkl"
    scaler_path = "ml_models/churn_scaler.pkl"
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logging.info("Loaded existing churn prediction model")
        else:
            # Create dummy model if none exists
            model, scaler = create_dummy_model('churn')
            logging.info("Created dummy churn prediction model")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading churn model: {str(e)}")
        # Fallback to creating a new dummy model
        model, scaler = create_dummy_model('churn')
        return model, scaler

def predict_churn(data):
    """
    Predict customer churn probability
    
    Args:
        data (dict): Input features for churn prediction
        
    Returns:
        dict: Prediction results with churn probability and risk level
    """
    try:
        model, scaler = get_model()
        
        # Expected features for churn prediction
        expected_features = [
            'tenure_months', 'monthly_charges', 'total_charges', 'contract_length',
            'payment_method_score', 'support_calls', 'usage_score', 'satisfaction_score'
        ]
        
        # Preprocess input data
        features = preprocess_features(data, expected_features)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get prediction and probability
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Calculate churn probability (probability of positive class)
        churn_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Determine risk level
        if churn_probability >= 0.8:
            risk_level = "Very High"
        elif churn_probability >= 0.6:
            risk_level = "High"
        elif churn_probability >= 0.4:
            risk_level = "Medium"
        elif churn_probability >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        # Calculate retention recommendation
        if churn_probability >= 0.6:
            recommendation = "Immediate intervention required"
        elif churn_probability >= 0.4:
            recommendation = "Monitor closely and engage"
        elif churn_probability >= 0.2:
            recommendation = "Standard retention activities"
        else:
            recommendation = "No immediate action needed"
        
        return {
            'churn_probability': round(churn_probability, 4),
            'confidence': round(churn_probability, 4),
            'risk_level': risk_level,
            'will_churn': bool(prediction),
            'recommendation': recommendation,
            'features_used': expected_features,
            'model_version': 'v1.0'
        }
        
    except Exception as e:
        logging.error(f"Churn prediction error: {str(e)}")
        # Return default/fallback prediction
        return {
            'churn_probability': 0.5,
            'confidence': 0.5,
            'risk_level': 'Unknown',
            'will_churn': False,
            'recommendation': 'Unable to determine',
            'error': str(e),
            'model_version': 'v1.0'
        }

def train_model(training_data):
    """
    Train a new churn prediction model with provided data
    
    Args:
        training_data (pd.DataFrame): Training dataset
        
    Returns:
        dict: Training results
    """
    try:
        # Prepare features and target
        feature_columns = [
            'tenure_months', 'monthly_charges', 'total_charges', 'contract_length',
            'payment_method_score', 'support_calls', 'usage_score', 'satisfaction_score'
        ]
        
        X = training_data[feature_columns]
        y = training_data['churned']  # Assuming this is the target column
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Save model and scaler
        os.makedirs('ml_models', exist_ok=True)
        joblib.dump(model, 'ml_models/churn_model.pkl')
        joblib.dump(scaler, 'ml_models/churn_scaler.pkl')
        
        # Calculate training score
        train_score = model.score(X_scaled, y)
        
        return {
            'message': 'Churn model trained successfully',
            'train_accuracy': round(train_score, 4),
            'features': feature_columns,
            'model_type': 'GradientBoostingClassifier'
        }
        
    except Exception as e:
        logging.error(f"Churn model training error: {str(e)}")
        return {
            'error': str(e),
            'message': 'Churn model training failed'
        }

def analyze_churn_factors(data):
    """Analyze factors contributing to churn risk"""
    try:
        model, scaler = get_model()
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'tenure_months', 'monthly_charges', 'total_charges', 'contract_length',
                'payment_method_score', 'support_calls', 'usage_score', 'satisfaction_score'
            ]
            
            # Preprocess input data
            features = preprocess_features(data, feature_names)
            
            # Get feature importance
            importances = model.feature_importances_
            
            # Calculate contribution of each feature to the prediction
            feature_contributions = []
            for i, (name, value, importance) in enumerate(zip(feature_names, features, importances)):
                contribution = value * importance
                feature_contributions.append({
                    'feature': name,
                    'value': round(value, 4),
                    'importance': round(importance, 4),
                    'contribution': round(contribution, 4)
                })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            return {
                'churn_factors': feature_contributions[:5],  # Top 5 factors
                'model_type': type(model).__name__
            }
        else:
            return {
                'error': 'Model does not support feature analysis',
                'model_type': type(model).__name__
            }
            
    except Exception as e:
        logging.error(f"Churn factor analysis error: {str(e)}")
        return {
            'error': str(e)
        }
