import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

def preprocess_features(data, expected_features):
    """
    Preprocess input data to match expected features
    
    Args:
        data (dict): Input data dictionary
        expected_features (list): List of expected feature names
        
    Returns:
        list: Preprocessed feature values
    """
    features = []
    
    for feature in expected_features:
        if feature in data:
            value = data[feature]
            
            # Handle different data types
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Try to convert string to number
                try:
                    features.append(float(value))
                except ValueError:
                    # For categorical strings, use hash-based encoding
                    features.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, bool):
                features.append(float(value))
            else:
                # Default value for unknown types
                features.append(0.0)
        else:
            # Use default values for missing features
            features.append(get_default_value(feature))
    
    return features

def get_default_value(feature_name):
    """
    Get default value for a feature based on its name
    
    Args:
        feature_name (str): Name of the feature
        
    Returns:
        float: Default value
    """
    # Mapping of feature patterns to default values
    defaults = {
        'score': 50.0,
        'rate': 0.5,
        'count': 0.0,
        'calls': 0.0,
        'size': 100.0,
        'budget': 10000.0,
        'charges': 100.0,
        'months': 12.0,
        'length': 12.0,
        'spend': 1000.0,
        'volume': 100.0,
        'factor': 1.0,
        'trend': 1.0,
        'indicator': 1.0,
        'activity': 0.5
    }
    
    feature_lower = feature_name.lower()
    
    for pattern, default_val in defaults.items():
        if pattern in feature_lower:
            return default_val
    
    # Generic default
    return 0.0

def create_dummy_model(model_type):
    """
    Create a realistic demo model that responds to input changes
    
    Args:
        model_type (str): Type of model to create
        
    Returns:
        tuple: (model, scaler) objects
    """
    try:
        logging.info(f"Creating realistic {model_type} model")
        
        # Create realistic training data that responds to inputs
        np.random.seed(42)
        
        if model_type == 'lead_scoring':
            # Create realistic lead scoring data
            n_samples = 2000
            
            # Generate realistic feature distributions
            company_size = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1])
            budget = np.random.exponential(50000, n_samples)  
            industry_score = np.random.normal(6, 2, n_samples)
            engagement_score = np.random.normal(5, 2, n_samples)
            demographic_score = np.random.normal(5, 2, n_samples)
            behavioral_score = np.random.normal(6, 2, n_samples)
            source_score = np.random.normal(5, 2, n_samples)
            
            # Clip scores to realistic ranges
            industry_score = np.clip(industry_score, 1, 10)
            engagement_score = np.clip(engagement_score, 1, 10)
            demographic_score = np.clip(demographic_score, 1, 10)
            behavioral_score = np.clip(behavioral_score, 1, 10)
            source_score = np.clip(source_score, 1, 10)
            
            X = np.column_stack([company_size, budget, industry_score, engagement_score, 
                               demographic_score, behavioral_score, source_score])
            
            # Create realistic lead quality based on logical business rules
            lead_quality = (
                company_size * 15 +  # Larger companies = better leads
                (budget / 10000) * 10 +  # Higher budget = better leads
                behavioral_score * 8 +  # Engagement is key
                industry_score * 6 +
                engagement_score * 5 +
                demographic_score * 4 +
                source_score * 3 +
                np.random.normal(0, 5, n_samples)  # Add some noise
            )
            
            # Convert to binary classification (good lead vs bad lead)
            y = (lead_quality > np.percentile(lead_quality, 60)).astype(int)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            
        elif model_type in ['churn', 'churn_prediction']:
            # Create realistic churn prediction data
            n_samples = 2000
            
            # Generate realistic customer features
            tenure_months = np.random.exponential(24, n_samples)
            monthly_charges = np.random.normal(65, 25, n_samples)
            total_charges = tenure_months * monthly_charges + np.random.normal(0, 500, n_samples)
            contract_length = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
            payment_method_score = np.random.normal(3, 1.5, n_samples)
            support_calls = np.random.poisson(2, n_samples)
            usage_score = np.random.normal(6, 2, n_samples)
            satisfaction_score = np.random.normal(6, 2, n_samples)
            
            # Clip scores to realistic ranges
            monthly_charges = np.clip(monthly_charges, 20, 150)
            payment_method_score = np.clip(payment_method_score, 1, 5)
            usage_score = np.clip(usage_score, 1, 10)
            satisfaction_score = np.clip(satisfaction_score, 1, 10)
            support_calls = np.clip(support_calls, 0, 15)
            
            X = np.column_stack([tenure_months, monthly_charges, total_charges, contract_length,
                               payment_method_score, support_calls, usage_score, satisfaction_score])
            
            # Create realistic churn probability based on business logic
            churn_risk = (
                -tenure_months * 0.02 +  # Longer tenure = lower churn
                monthly_charges * 0.01 +  # Higher charges = higher churn
                support_calls * 5 +  # More support calls = higher churn
                (5 - contract_length) * 10 +  # Month-to-month = higher churn
                (5 - payment_method_score) * 8 +  # Poor payment method = higher churn
                (10 - satisfaction_score) * 6 +  # Low satisfaction = higher churn
                (10 - usage_score) * 4 +  # Low usage = higher churn
                np.random.normal(0, 8, n_samples)  # Add noise
            )
            
            # Convert to binary classification
            y = (churn_risk > np.percentile(churn_risk, 70)).astype(int)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
            
        elif model_type == 'sales_forecast':
            # Create realistic sales forecasting data
            n_samples = 2000
            
            # Generate realistic sales features
            historical_sales = np.random.exponential(100000, n_samples)
            marketing_spend = historical_sales * np.random.uniform(0.05, 0.15, n_samples)
            seasonality = np.random.normal(1.0, 0.3, n_samples)
            economic_indicators = np.random.normal(1.0, 0.2, n_samples)
            product_category = np.random.choice([1, 2, 3, 4, 5], n_samples)
            market_growth = np.random.normal(5, 10, n_samples)
            competitive_index = np.random.normal(6, 2, n_samples)
            
            # Clip to realistic ranges
            seasonality = np.clip(seasonality, 0.5, 2.0)
            economic_indicators = np.clip(economic_indicators, 0.7, 1.5)
            competitive_index = np.clip(competitive_index, 1, 10)
            market_growth = np.clip(market_growth, -20, 30)
            
            X = np.column_stack([historical_sales, marketing_spend, seasonality, economic_indicators,
                               product_category, market_growth, competitive_index, 
                               np.ones(n_samples)])  # Add intercept feature
            
            # Create realistic forecast based on business logic
            forecast = (
                historical_sales * seasonality * economic_indicators *  # Base forecast
                (1 + market_growth / 100) *  # Market growth factor
                (marketing_spend / historical_sales * 2 + 0.8) *  # Marketing impact
                (competitive_index / 10 * 0.2 + 0.9) +  # Competition factor
                np.random.normal(0, historical_sales * 0.1, n_samples)  # Noise
            )
            
            y = np.maximum(forecast, historical_sales * 0.3)  # Minimum realistic forecast
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model.fit(X_scaled, y)
        
        # Save models
        os.makedirs('ml_models', exist_ok=True)
        joblib.dump(model, f'ml_models/{model_type}_model.pkl')
        joblib.dump(scaler, f'ml_models/{model_type}_scaler.pkl')
        
        logging.info(f"Realistic {model_type} model created and saved")
        
        return model, scaler
        
    except Exception as e:
        logging.error(f"Error creating {model_type} model: {str(e)}")
        raise

def validate_input_data(data, required_fields=None):
    """
    Validate input data for ML predictions
    
    Args:
        data (dict): Input data to validate
        required_fields (list): List of required field names
        
    Returns:
        dict: Validation results
    """
    if not isinstance(data, dict):
        return {
            'valid': False,
            'error': 'Input data must be a dictionary'
        }
    
    if not data:
        return {
            'valid': False,
            'error': 'Input data is empty'
        }
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'missing_fields': [],
        'invalid_fields': []
    }
    
    # Check required fields
    if required_fields:
        for field in required_fields:
            if field not in data:
                validation_results['missing_fields'].append(field)
    
    # Check data types and values
    for key, value in data.items():
        if value is None:
            validation_results['warnings'].append(f"Field '{key}' is None")
        elif isinstance(value, str) and value.strip() == "":
            validation_results['warnings'].append(f"Field '{key}' is empty string")
        elif isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
            validation_results['invalid_fields'].append(f"Field '{key}' has invalid numeric value")
    
    # Update validation status
    if validation_results['missing_fields'] or validation_results['invalid_fields']:
        validation_results['valid'] = False
        
        error_parts = []
        if validation_results['missing_fields']:
            error_parts.append(f"Missing fields: {', '.join(validation_results['missing_fields'])}")
        if validation_results['invalid_fields']:
            error_parts.append(f"Invalid fields: {', '.join(validation_results['invalid_fields'])}")
        
        validation_results['error'] = '; '.join(error_parts)
    
    return validation_results

def calculate_model_metrics(y_true, y_pred, model_type='classification'):
    """
    Calculate standard metrics for model evaluation
    
    Args:
        y_true (array): True target values
        y_pred (array): Predicted values
        model_type (str): 'classification' or 'regression'
        
    Returns:
        dict: Model metrics
    """
    try:
        if model_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            }
            
        elif model_type == 'regression':
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mean_absolute_error': round(mae, 4),
                'mean_squared_error': round(mse, 4),
                'root_mean_squared_error': round(rmse, 4),
                'r2_score': round(r2, 4)
            }
        
        else:
            return {'error': f"Unknown model type: {model_type}"}
            
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return {'error': str(e)}

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method
    
    Args:
        data (array-like): Data to normalize
        method (str): Normalization method ('minmax', 'standard', 'robust')
        
    Returns:
        array: Normalized data
    """
    try:
        data = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)
            
        elif method == 'standard':
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return np.zeros_like(data)
            return (data - mean_val) / std_val
            
        elif method == 'robust':
            median_val = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(data)
            return (data - median_val) / iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    except Exception as e:
        logging.error(f"Error normalizing data: {str(e)}")
        return data

def encode_categorical_features(data, feature_mapping=None):
    """
    Encode categorical features for ML models
    
    Args:
        data (dict): Input data with categorical features
        feature_mapping (dict): Mapping of categorical values to numeric
        
    Returns:
        dict: Data with encoded categorical features
    """
    encoded_data = data.copy()
    
    # Default mappings for common categorical features
    default_mappings = {
        'industry': {
            'technology': 5, 'finance': 4, 'healthcare': 4, 'retail': 3,
            'manufacturing': 3, 'education': 2, 'other': 1
        },
        'company_size': {
            'enterprise': 5, 'large': 4, 'medium': 3, 'small': 2, 'startup': 1
        },
        'source': {
            'referral': 5, 'organic': 4, 'paid_search': 3, 'social': 2, 'email': 1
        },
        'contract_type': {
            'annual': 3, 'quarterly': 2, 'monthly': 1
        },
        'payment_method': {
            'automatic': 3, 'credit_card': 2, 'bank_transfer': 1, 'check': 0
        }
    }
    
    # Use provided mapping or default
    mappings = feature_mapping or default_mappings
    
    for feature, value in encoded_data.items():
        if isinstance(value, str):
            # Check if we have a mapping for this feature
            feature_lower = feature.lower()
            for map_key, mapping in mappings.items():
                if map_key in feature_lower:
                    encoded_value = mapping.get(value.lower(), 0)
                    encoded_data[feature] = encoded_value
                    break
    
    return encoded_data
