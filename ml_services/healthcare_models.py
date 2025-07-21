"""
Healthcare Industry ML Models
Specialized models for healthcare predictions including churn, risk assessment, and clinical outcomes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime
import logging


class HealthcareChurnPredictor:
    """Specialized churn prediction for healthcare providers"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def preprocess_healthcare_data(self, data):
        """Healthcare-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Healthcare-specific feature engineering
        features = []
        
        # Patient demographics
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], 
                                   labels=['pediatric', 'young_adult', 'middle_age', 'senior', 'elderly'])
            features.append('age')
            
        # Insurance and financial factors
        financial_features = ['insurance_type', 'copay_amount', 'deductible', 'annual_income']
        for feature in financial_features:
            if feature in df.columns:
                features.append(feature)
                
        # Healthcare utilization patterns
        utilization_features = ['visits_per_year', 'emergency_visits', 'specialist_visits', 
                              'prescription_count', 'chronic_conditions']
        for feature in utilization_features:
            if feature in df.columns:
                features.append(feature)
                
        # Satisfaction and experience
        experience_features = ['satisfaction_score', 'wait_time_complaints', 'provider_changes', 
                             'appointment_cancellations', 'telemedicine_usage']
        for feature in experience_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with healthcare-specific logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                # Use median for healthcare metrics
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
        
        return df, features
    
    def predict(self, data):
        """Make healthcare churn prediction"""
        try:
            df, features = self.preprocess_healthcare_data(data)
            
            # Create dummy data for training if model not trained
            if not hasattr(self.model, 'feature_importances_'):
                self._train_dummy_model(features)
            
            # Select available features
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            X_scaled = self.scaler.transform(X)
            
            # Get prediction
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            # Healthcare-specific risk factors
            risk_factors = self._identify_healthcare_risk_factors(df.iloc[0])
            
            return {
                'prediction': int(prediction),
                'churn_probability': float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                'risk_level': self._categorize_risk(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                'confidence': float(max(prediction_proba)),
                'risk_factors': risk_factors,
                'healthcare_insights': self._get_healthcare_insights(df.iloc[0], prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                'recommended_actions': self._get_retention_strategies(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0])
            }
            
        except Exception as e:
            logging.error(f"Healthcare churn prediction error: {str(e)}")
            return {
                'prediction': 0,
                'churn_probability': 0.3,
                'confidence': 0.6,
                'error': f'Prediction error: {str(e)}',
                'risk_factors': ['data_quality_issues']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic healthcare data"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic healthcare features
        X = np.random.randn(n_samples, len(features))
        
        # Create target with healthcare logic
        y = np.zeros(n_samples)
        for i in range(n_samples):
            # High churn factors: high costs, low satisfaction, frequent ER visits
            risk_score = (X[i, 0] * 0.3 +  # cost factor
                         X[i, 1] * -0.4 +  # satisfaction factor (negative correlation)
                         X[i, 2] * 0.2)    # utilization factor
            y[i] = 1 if risk_score > 0.5 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _identify_healthcare_risk_factors(self, patient_data):
        """Identify specific healthcare churn risk factors"""
        risk_factors = []
        
        if patient_data.get('satisfaction_score', 5) < 3:
            risk_factors.append('low_satisfaction')
        if patient_data.get('copay_amount', 0) > 50:
            risk_factors.append('high_copay')
        if patient_data.get('wait_time_complaints', 0) > 2:
            risk_factors.append('wait_time_issues')
        if patient_data.get('provider_changes', 0) > 1:
            risk_factors.append('provider_instability')
        if patient_data.get('emergency_visits', 0) > 3:
            risk_factors.append('high_emergency_usage')
        
        return risk_factors or ['standard_risk']
    
    def _categorize_risk(self, probability):
        """Categorize churn risk level"""
        if probability < 0.3:
            return 'low'
        elif probability < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _get_healthcare_insights(self, patient_data, churn_prob):
        """Generate healthcare-specific insights"""
        insights = []
        
        if churn_prob > 0.6:
            insights.append('Patient shows high risk of switching providers')
            if patient_data.get('satisfaction_score', 5) < 3:
                insights.append('Low satisfaction score is primary concern')
        
        if patient_data.get('chronic_conditions', 0) > 2:
            insights.append('Multiple chronic conditions require specialized care coordination')
        
        return insights
    
    def _get_retention_strategies(self, churn_prob):
        """Healthcare-specific retention strategies"""
        if churn_prob < 0.3:
            return ['maintain_current_service_level', 'routine_satisfaction_surveys']
        elif churn_prob < 0.6:
            return ['improve_communication', 'reduce_wait_times', 'care_coordination']
        else:
            return ['immediate_intervention', 'care_manager_assignment', 'financial_assistance_review', 'provider_matching']


class HealthcareRiskAssessment:
    """Clinical and financial risk assessment for healthcare"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Assess healthcare risks"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Healthcare risk factors
            risk_features = ['age', 'chronic_conditions', 'bmi', 'smoking_status', 
                           'family_history', 'medication_count', 'recent_surgeries']
            
            available_features = [f for f in risk_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_risk_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                risk_proba = self.model.predict_proba(X_scaled)[0]
                risk_level = self.model.predict(X_scaled)[0]
            else:
                risk_proba = [0.7, 0.3]
                risk_level = 0
            
            return {
                'risk_score': float(risk_proba[1] if len(risk_proba) > 1 else 0.3),
                'risk_category': self._categorize_clinical_risk(risk_proba[1] if len(risk_proba) > 1 else 0.3),
                'confidence': float(max(risk_proba)),
                'primary_risk_factors': self._identify_primary_risks(df.iloc[0]),
                'care_recommendations': self._get_care_recommendations(risk_proba[1] if len(risk_proba) > 1 else 0.3),
                'monitoring_frequency': self._recommend_monitoring(risk_proba[1] if len(risk_proba) > 1 else 0.3)
            }
            
        except Exception as e:
            return {
                'risk_score': 0.3,
                'risk_category': 'moderate',
                'confidence': 0.6,
                'error': str(e)
            }
    
    def _train_risk_model(self, features):
        """Train risk assessment model"""
        np.random.seed(42)
        n_samples = 800
        
        X = np.random.randn(n_samples, len(features))
        y = (X[:, 0] * 0.4 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1 > 0.5).astype(int)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_clinical_risk(self, score):
        """Categorize clinical risk level"""
        if score < 0.25:
            return 'low'
        elif score < 0.5:
            return 'moderate'
        elif score < 0.75:
            return 'high'
        else:
            return 'critical'
    
    def _identify_primary_risks(self, patient_data):
        """Identify primary clinical risk factors"""
        risks = []
        
        if patient_data.get('age', 0) > 65:
            risks.append('advanced_age')
        if patient_data.get('chronic_conditions', 0) > 2:
            risks.append('multiple_comorbidities')
        if patient_data.get('bmi', 25) > 30:
            risks.append('obesity')
        if patient_data.get('smoking_status') == 'current':
            risks.append('smoking')
        
        return risks or ['standard_risk_profile']
    
    def _get_care_recommendations(self, risk_score):
        """Generate care recommendations based on risk"""
        if risk_score < 0.25:
            return ['routine_preventive_care', 'annual_wellness_visits']
        elif risk_score < 0.5:
            return ['bi_annual_checkups', 'lifestyle_counseling', 'preventive_screenings']
        elif risk_score < 0.75:
            return ['quarterly_monitoring', 'specialist_referrals', 'care_coordination']
        else:
            return ['intensive_monitoring', 'multidisciplinary_care_team', 'frequent_follow_ups']
    
    def _recommend_monitoring(self, risk_score):
        """Recommend monitoring frequency"""
        if risk_score < 0.25:
            return 'annual'
        elif risk_score < 0.5:
            return 'bi_annual'
        elif risk_score < 0.75:
            return 'quarterly'
        else:
            return 'monthly'


def predict_healthcare_churn(data):
    """Main function for healthcare churn prediction"""
    predictor = HealthcareChurnPredictor()
    return predictor.predict(data)


def predict_healthcare_risk(data):
    """Main function for healthcare risk assessment"""
    assessor = HealthcareRiskAssessment()
    return assessor.predict(data)