import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import logging
from datetime import datetime, timedelta
from .utils import preprocess_features, create_dummy_model

def get_model():
    """Load or create sales forecasting model"""
    model_path = "ml_models/sales_forecast_model.pkl"
    scaler_path = "ml_models/sales_forecast_scaler.pkl"
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logging.info("Loaded existing sales forecast model")
        else:
            # Create dummy model if none exists
            model, scaler = create_dummy_model('sales_forecast')
            logging.info("Created dummy sales forecast model")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading sales forecast model: {str(e)}")
        # Fallback to creating a new dummy model
        model, scaler = create_dummy_model('sales_forecast')
        return model, scaler

def predict_sales(data):
    """
    Predict sales forecast based on input data
    
    Args:
        data (dict): Input features for sales forecasting
        
    Returns:
        dict: Prediction results with forecast and confidence intervals
    """
    try:
        model, scaler = get_model()
        
        # Expected features for sales forecasting
        expected_features = [
            'historical_sales_avg', 'seasonality_factor', 'market_trend', 'marketing_spend',
            'lead_volume', 'conversion_rate', 'economic_indicator', 'competitor_activity'
        ]
        
        # Preprocess input data
        features = preprocess_features(data, expected_features)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get prediction
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence intervals (approximate using model variance if available)
        if hasattr(model, 'estimators_'):
            # For ensemble models, use prediction variance
            individual_predictions = [tree.predict(features_scaled)[0] for tree in model.estimators_]
            prediction_std = np.std(individual_predictions)
            confidence_interval = 1.96 * prediction_std  # 95% confidence interval
        else:
            # Fallback confidence interval (10% of prediction)
            confidence_interval = abs(prediction) * 0.1
        
        # Calculate forecast range
        lower_bound = max(0, prediction - confidence_interval)
        upper_bound = prediction + confidence_interval
        
        # Determine forecast quality
        coefficient_of_variation = confidence_interval / abs(prediction) if prediction != 0 else 0
        
        if coefficient_of_variation <= 0.1:
            forecast_quality = "High"
        elif coefficient_of_variation <= 0.2:
            forecast_quality = "Medium"
        else:
            forecast_quality = "Low"
        
        # Get forecast period
        forecast_period = data.get('forecast_period', 'monthly')
        
        return {
            'forecast': round(prediction, 2),
            'confidence': round(1 - coefficient_of_variation, 4),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'confidence_interval': round(confidence_interval, 2),
            'forecast_quality': forecast_quality,
            'forecast_period': forecast_period,
            'features_used': expected_features,
            'model_version': 'v1.0'
        }
        
    except Exception as e:
        logging.error(f"Sales forecast prediction error: {str(e)}")
        # Return default/fallback prediction
        historical_avg = data.get('historical_sales_avg', 10000)
        return {
            'forecast': historical_avg,
            'confidence': 0.5,
            'lower_bound': historical_avg * 0.8,
            'upper_bound': historical_avg * 1.2,
            'confidence_interval': historical_avg * 0.2,
            'forecast_quality': 'Unknown',
            'forecast_period': 'monthly',
            'error': str(e),
            'model_version': 'v1.0'
        }

def predict_sales_trend(data, periods=12):
    """
    Predict sales trend for multiple periods
    
    Args:
        data (dict): Base input features
        periods (int): Number of periods to forecast
        
    Returns:
        dict: Multi-period forecast results
    """
    try:
        model, scaler = get_model()
        
        forecasts = []
        current_data = data.copy()
        
        for period in range(1, periods + 1):
            # Adjust features for future periods (simple trend adjustment)
            if 'seasonality_factor' in current_data:
                # Apply seasonal adjustment (simplified)
                month = (datetime.now().month + period - 1) % 12 + 1
                seasonal_multiplier = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                current_data['seasonality_factor'] *= seasonal_multiplier
            
            if 'market_trend' in current_data:
                # Apply trend growth (simplified 2% monthly growth)
                current_data['market_trend'] *= 1.02
            
            # Get prediction for this period
            result = predict_sales(current_data)
            
            forecasts.append({
                'period': period,
                'forecast': result['forecast'],
                'confidence': result['confidence'],
                'lower_bound': result['lower_bound'],
                'upper_bound': result['upper_bound']
            })
            
            # Update historical average for next iteration
            if 'historical_sales_avg' in current_data:
                current_data['historical_sales_avg'] = result['forecast']
        
        # Calculate trend statistics
        forecast_values = [f['forecast'] for f in forecasts]
        trend_slope = np.polyfit(range(len(forecast_values)), forecast_values, 1)[0]
        
        if trend_slope > 0:
            trend_direction = "Growing"
        elif trend_slope < 0:
            trend_direction = "Declining"
        else:
            trend_direction = "Stable"
        
        return {
            'forecasts': forecasts,
            'trend_direction': trend_direction,
            'trend_slope': round(trend_slope, 2),
            'total_forecast': round(sum(forecast_values), 2),
            'average_confidence': round(np.mean([f['confidence'] for f in forecasts]), 4),
            'periods': periods,
            'model_version': 'v1.0'
        }
        
    except Exception as e:
        logging.error(f"Sales trend prediction error: {str(e)}")
        return {
            'error': str(e),
            'model_version': 'v1.0'
        }

def train_model(training_data):
    """
    Train a new sales forecasting model with provided data
    
    Args:
        training_data (pd.DataFrame): Training dataset
        
    Returns:
        dict: Training results
    """
    try:
        # Prepare features and target
        feature_columns = [
            'historical_sales_avg', 'seasonality_factor', 'market_trend', 'marketing_spend',
            'lead_volume', 'conversion_rate', 'economic_indicator', 'competitor_activity'
        ]
        
        X = training_data[feature_columns]
        y = training_data['actual_sales']  # Assuming this is the target column
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Save model and scaler
        os.makedirs('ml_models', exist_ok=True)
        joblib.dump(model, 'ml_models/sales_forecast_model.pkl')
        joblib.dump(scaler, 'ml_models/sales_forecast_scaler.pkl')
        
        # Calculate training metrics
        train_predictions = model.predict(X_scaled)
        train_score = model.score(X_scaled, y)
        mae = mean_absolute_error(y, train_predictions)
        rmse = np.sqrt(mean_squared_error(y, train_predictions))
        
        return {
            'message': 'Sales forecast model trained successfully',
            'train_r2_score': round(train_score, 4),
            'mean_absolute_error': round(mae, 2),
            'root_mean_squared_error': round(rmse, 2),
            'features': feature_columns,
            'model_type': 'RandomForestRegressor'
        }
        
    except Exception as e:
        logging.error(f"Sales forecast model training error: {str(e)}")
        return {
            'error': str(e),
            'message': 'Sales forecast model training failed'
        }

def get_feature_importance():
    """Get feature importance from the trained sales forecast model"""
    try:
        model, _ = get_model()
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'historical_sales_avg', 'seasonality_factor', 'market_trend', 'marketing_spend',
                'lead_volume', 'conversion_rate', 'economic_indicator', 'competitor_activity'
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
