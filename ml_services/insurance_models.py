"""
Insurance Industry ML Models
Specialized models for insurance including risk assessment, claim prediction, and fraud detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
import logging


class InsuranceRiskAssessment:
    """Comprehensive risk assessment for insurance underwriting"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_insurance_data(self, data):
        """Insurance-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Insurance feature engineering
        features = []
        
        # Personal demographics
        demographic_features = ['age', 'gender', 'marital_status', 'occupation', 'income_annual',
                              'education_level', 'dependents_count', 'geographic_region']
        for feature in demographic_features:
            if feature in df.columns:
                features.append(feature)
                
        # Health and lifestyle (for health/life insurance)
        health_features = ['bmi', 'smoking_status', 'alcohol_consumption', 'exercise_frequency',
                         'medical_history_score', 'family_medical_history', 'prescription_count',
                         'hospital_visits_yearly', 'chronic_conditions_count']
        for feature in health_features:
            if feature in df.columns:
                features.append(feature)
                
        # Property details (for property insurance)
        property_features = ['property_value', 'property_age', 'construction_type', 'location_risk_score',
                           'security_features_score', 'natural_disaster_risk', 'crime_rate_area',
                           'fire_station_distance', 'previous_claims_count']
        for feature in property_features:
            if feature in df.columns:
                features.append(feature)
        
        # Vehicle details (for auto insurance)
        vehicle_features = ['vehicle_age', 'vehicle_value', 'safety_rating', 'theft_risk_score',
                          'annual_mileage', 'primary_use', 'driver_experience_years', 'accident_history_count']
        for feature in vehicle_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with insurance-specific logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                if 'score' in col.lower() or 'rating' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 3)
                elif 'count' in col.lower():
                    df[col] = df[col].fillna(0)  # No history = 0 count
                elif 'age' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 35)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Assess insurance risk"""
        try:
            df, features = self.preprocess_insurance_data(data)
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_dummy_model(features)
            
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            X_scaled = self.scaler.transform(X)
            risk_score = self.model.predict(X_scaled)[0]
            
            # Normalize risk score to 0-100 scale
            risk_score = max(0, min(100, risk_score))
            
            return {
                'risk_score': float(risk_score),
                'risk_category': self._categorize_risk(risk_score),
                'premium_multiplier': self._calculate_premium_multiplier(risk_score),
                'risk_factors': self._identify_risk_factors(df.iloc[0]),
                'underwriting_recommendation': self._recommend_underwriting_action(risk_score),
                'mitigation_suggestions': self._suggest_risk_mitigation(df.iloc[0]),
                'coverage_recommendations': self._recommend_coverage_options(risk_score, df.iloc[0]),
                'monitoring_requirements': self._determine_monitoring_needs(risk_score)
            }
            
        except Exception as e:
            logging.error(f"Insurance risk assessment error: {str(e)}")
            return {
                'risk_score': 50,
                'risk_category': 'moderate',
                'premium_multiplier': 1.0,
                'error': f'Risk assessment error: {str(e)}',
                'risk_factors': ['data_processing_error']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic insurance data"""
        np.random.seed(42)
        n_samples = 2000
        
        X = np.random.randn(n_samples, len(features))
        
        # Risk logic: age, history, lifestyle, property factors
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_risk = 50
            age_factor = abs(X[i, 0] - 1) * 10 if len(features) > 0 else 0  # Higher risk for very young/old
            history_factor = X[i, 1] * 15 if len(features) > 1 else 0      # Claims history
            lifestyle_factor = X[i, 2] * 12 if len(features) > 2 else 0    # Health/lifestyle risks
            
            y[i] = max(0, min(100, base_risk + age_factor + history_factor + lifestyle_factor + np.random.randn() * 5))
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_risk(self, score):
        """Categorize risk level"""
        if score <= 25:
            return 'low'
        elif score <= 50:
            return 'moderate'
        elif score <= 75:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_premium_multiplier(self, risk_score):
        """Calculate premium multiplier based on risk"""
        base_multiplier = 1.0
        risk_adjustment = (risk_score - 50) / 100  # Adjust from baseline of 50
        return max(0.5, base_multiplier + risk_adjustment)
    
    def _identify_risk_factors(self, applicant_data):
        """Identify specific risk factors"""
        risks = []
        
        if applicant_data.get('age', 35) < 25 or applicant_data.get('age', 35) > 65:
            risks.append('age_risk_factor')
        if applicant_data.get('smoking_status', 'non_smoker') == 'smoker':
            risks.append('smoking_risk')
        if applicant_data.get('previous_claims_count', 0) > 2:
            risks.append('claims_history')
        if applicant_data.get('bmi', 25) > 30:
            risks.append('health_risk_obesity')
        if applicant_data.get('chronic_conditions_count', 0) > 1:
            risks.append('multiple_health_conditions')
        if applicant_data.get('location_risk_score', 3) > 7:
            risks.append('high_risk_location')
        if applicant_data.get('accident_history_count', 0) > 1:
            risks.append('driving_history')
        
        return risks or ['standard_risk_profile']
    
    def _recommend_underwriting_action(self, risk_score):
        """Recommend underwriting decision"""
        if risk_score <= 30:
            return 'auto_approve_preferred_rates'
        elif risk_score <= 60:
            return 'approve_standard_rates'
        elif risk_score <= 80:
            return 'approve_with_conditions'
        else:
            return 'manual_underwriting_required'
    
    def _suggest_risk_mitigation(self, applicant_data):
        """Suggest risk mitigation strategies"""
        suggestions = []
        
        if applicant_data.get('smoking_status', 'non_smoker') == 'smoker':
            suggestions.append('smoking_cessation_program_discount')
        if applicant_data.get('security_features_score', 5) < 3:
            suggestions.append('install_security_system')
        if applicant_data.get('safety_rating', 5) < 4:
            suggestions.append('consider_safer_vehicle')
        if applicant_data.get('exercise_frequency', 3) < 2:
            suggestions.append('wellness_program_participation')
        
        return suggestions or ['no_specific_mitigation_needed']
    
    def _recommend_coverage_options(self, risk_score, applicant_data):
        """Recommend appropriate coverage options"""
        if risk_score <= 30:
            return ['comprehensive_coverage', 'optional_riders', 'loyalty_discounts']
        elif risk_score <= 60:
            return ['standard_coverage', 'deductible_options']
        else:
            return ['basic_coverage', 'higher_deductibles', 'exclusions_may_apply']
    
    def _determine_monitoring_needs(self, risk_score):
        """Determine ongoing monitoring requirements"""
        if risk_score <= 40:
            return 'annual_review'
        elif risk_score <= 70:
            return 'semi_annual_review'
        else:
            return 'quarterly_monitoring'


class InsuranceClaimPredictor:
    """Predict likelihood and severity of insurance claims"""
    
    def __init__(self):
        self.claim_likelihood_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        self.claim_severity_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Predict claim likelihood and severity"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Claim prediction features
            claim_features = ['policy_age_months', 'premium_paid_total', 'coverage_amount', 'deductible_amount',
                            'policyholder_age', 'claim_history_count', 'risk_score', 'policy_type',
                            'geographic_claims_frequency', 'seasonal_factor', 'economic_index']
            
            available_features = [f for f in claim_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy models if needed
            if not hasattr(self.claim_likelihood_model, 'feature_importances_'):
                self._train_claim_models(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                
                # Predict claim likelihood
                claim_proba = self.claim_likelihood_model.predict_proba(X_scaled)[0]
                claim_likelihood = claim_proba[1] if len(claim_proba) > 1 else 0.15
                
                # Predict claim severity
                claim_severity = self.claim_severity_model.predict(X_scaled)[0]
            else:
                claim_likelihood = 0.15
                claim_severity = 5000
            
            return {
                'claim_probability': float(claim_likelihood),
                'expected_claim_amount': float(max(0, claim_severity)),
                'claim_risk_level': self._categorize_claim_risk(claim_likelihood),
                'expected_annual_claims': self._calculate_annual_claims(claim_likelihood, df.iloc[0]),
                'claim_triggers': self._identify_claim_triggers(df.iloc[0]),
                'prevention_recommendations': self._recommend_claim_prevention(df.iloc[0]),
                'reserve_recommendation': self._calculate_reserve_needs(claim_likelihood, claim_severity),
                'pricing_impact': self._assess_pricing_impact(claim_likelihood, claim_severity)
            }
            
        except Exception as e:
            return {
                'claim_probability': 0.15,
                'expected_claim_amount': 5000,
                'claim_risk_level': 'moderate',
                'error': str(e)
            }
    
    def _train_claim_models(self, features):
        """Train claim prediction models"""
        np.random.seed(42)
        n_samples = 1500
        
        X = np.random.randn(n_samples, len(features))
        
        # Claim likelihood logic
        y_likelihood = np.zeros(n_samples)
        for i in range(n_samples):
            claim_score = (X[i, 0] * 0.3 +      # policy age
                          X[i, 1] * 0.4 +       # history
                          X[i, 2] * 0.2)        # risk factors
            y_likelihood[i] = 1 if claim_score > 0.2 else 0
        
        # Claim severity logic
        y_severity = np.zeros(n_samples)
        for i in range(n_samples):
            base_amount = 5000
            severity_factor = X[i, 0] * 3000 + X[i, 1] * 2000 + np.random.randn() * 1000
            y_severity[i] = max(100, base_amount + severity_factor)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.claim_likelihood_model.fit(X_scaled, y_likelihood)
        self.claim_severity_model.fit(X_scaled, y_severity)
    
    def _categorize_claim_risk(self, probability):
        """Categorize claim risk level"""
        if probability < 0.1:
            return 'low'
        elif probability < 0.25:
            return 'moderate'
        elif probability < 0.5:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_annual_claims(self, claim_probability, policy_data):
        """Calculate expected annual claims"""
        base_frequency = claim_probability * 1.2  # Adjust for annual frequency
        return float(base_frequency)
    
    def _identify_claim_triggers(self, policy_data):
        """Identify potential claim triggers"""
        triggers = []
        
        if policy_data.get('claim_history_count', 0) > 2:
            triggers.append('previous_claims_pattern')
        if policy_data.get('geographic_claims_frequency', 0.1) > 0.3:
            triggers.append('high_risk_location')
        if policy_data.get('seasonal_factor', 1.0) > 1.5:
            triggers.append('seasonal_risk_elevation')
        if policy_data.get('policy_age_months', 12) < 6:
            triggers.append('new_policy_risk')
        
        return triggers or ['standard_risk_factors']
    
    def _recommend_claim_prevention(self, policy_data):
        """Recommend claim prevention strategies"""
        recommendations = []
        
        if policy_data.get('risk_score', 50) > 70:
            recommendations.append('risk_assessment_consultation')
        if policy_data.get('deductible_amount', 1000) < 500:
            recommendations.append('consider_higher_deductible')
        
        recommendations.append('regular_maintenance_reminders')
        recommendations.append('safety_education_programs')
        
        return recommendations
    
    def _calculate_reserve_needs(self, claim_likelihood, claim_severity):
        """Calculate recommended claim reserves"""
        expected_cost = claim_likelihood * claim_severity
        reserve_multiplier = 1.5  # Safety factor
        
        return {
            'recommended_reserve': float(expected_cost * reserve_multiplier),
            'confidence_interval_upper': float(expected_cost * 2.0),
            'confidence_interval_lower': float(expected_cost * 0.5)
        }
    
    def _assess_pricing_impact(self, claim_likelihood, claim_severity):
        """Assess impact on pricing strategy"""
        expected_cost = claim_likelihood * claim_severity
        
        if expected_cost > 10000:
            return 'significant_premium_increase_recommended'
        elif expected_cost > 5000:
            return 'moderate_premium_adjustment'
        else:
            return 'maintain_current_pricing'


class InsuranceFraudDetection:
    """Detect potentially fraudulent insurance claims"""
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Detect potential insurance fraud"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Fraud detection features
            fraud_features = ['claim_amount', 'time_since_policy_start', 'claim_submission_delay',
                            'medical_provider_risk_score', 'claimant_history_score', 'claim_complexity_score',
                            'documentation_completeness', 'witness_availability', 'injury_severity_vs_incident']
            
            available_features = [f for f in fraud_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy models if needed
            if not hasattr(self.classification_model, 'feature_importances_'):
                self._train_fraud_models(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                
                # Anomaly detection
                anomaly_score = self.model.decision_function(X_scaled)[0]
                is_anomaly = self.model.predict(X_scaled)[0] == -1
                
                # Classification prediction
                fraud_proba = self.classification_model.predict_proba(X_scaled)[0]
                fraud_prediction = self.classification_model.predict(X_scaled)[0]
            else:
                anomaly_score = 0.1
                is_anomaly = False
                fraud_proba = [0.9, 0.1]
                fraud_prediction = 0
            
            fraud_probability = fraud_proba[1] if len(fraud_proba) > 1 else 0.1
            
            return {
                'fraud_probability': float(fraud_probability),
                'fraud_risk_level': self._categorize_fraud_risk(fraud_probability),
                'anomaly_detected': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'fraud_indicators': self._identify_fraud_indicators(df.iloc[0]),
                'investigation_priority': self._determine_investigation_priority(fraud_probability),
                'recommended_actions': self._recommend_fraud_actions(fraud_probability),
                'additional_verification_needed': self._suggest_verification_steps(df.iloc[0])
            }
            
        except Exception as e:
            return {
                'fraud_probability': 0.1,
                'fraud_risk_level': 'low',
                'anomaly_detected': False,
                'error': str(e)
            }
    
    def _train_fraud_models(self, features):
        """Train fraud detection models"""
        np.random.seed(42)
        n_samples = 2000
        
        X = np.random.randn(n_samples, len(features))
        
        # Fraud classification logic
        y_fraud = np.zeros(n_samples)
        for i in range(n_samples):
            fraud_score = (abs(X[i, 0]) * 0.4 +    # unusual claim amount
                          abs(X[i, 1]) * 0.3 +      # timing anomalies
                          abs(X[i, 2]) * 0.2)       # documentation issues
            y_fraud[i] = 1 if fraud_score > 1.2 else 0
        
        # Ensure reasonable fraud rate (10%)
        fraud_count = int(n_samples * 0.1)
        fraud_indices = np.random.choice(n_samples, fraud_count, replace=False)
        y_fraud[fraud_indices] = 1
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model.fit(X_scaled)
        self.classification_model.fit(X_scaled, y_fraud)
    
    def _categorize_fraud_risk(self, probability):
        """Categorize fraud risk level"""
        if probability < 0.15:
            return 'low'
        elif probability < 0.4:
            return 'moderate'
        elif probability < 0.7:
            return 'high'
        else:
            return 'critical'
    
    def _identify_fraud_indicators(self, claim_data):
        """Identify specific fraud indicators"""
        indicators = []
        
        if claim_data.get('claim_amount', 1000) > 50000:
            indicators.append('unusually_high_claim_amount')
        if claim_data.get('time_since_policy_start', 365) < 30:
            indicators.append('claim_shortly_after_policy_start')
        if claim_data.get('claim_submission_delay', 1) > 30:
            indicators.append('delayed_claim_reporting')
        if claim_data.get('documentation_completeness', 0.9) < 0.6:
            indicators.append('incomplete_documentation')
        if claim_data.get('medical_provider_risk_score', 3) > 7:
            indicators.append('high_risk_medical_provider')
        if claim_data.get('witness_availability', 1) == 0:
            indicators.append('no_independent_witnesses')
        
        return indicators or ['no_significant_fraud_indicators']
    
    def _determine_investigation_priority(self, fraud_probability):
        """Determine investigation priority level"""
        if fraud_probability > 0.7:
            return 'urgent_investigation'
        elif fraud_probability > 0.4:
            return 'priority_review'
        elif fraud_probability > 0.2:
            return 'routine_verification'
        else:
            return 'standard_processing'
    
    def _recommend_fraud_actions(self, fraud_probability):
        """Recommend actions based on fraud risk"""
        if fraud_probability > 0.7:
            return ['suspend_payment', 'special_investigation_unit', 'external_investigation']
        elif fraud_probability > 0.4:
            return ['detailed_review', 'additional_documentation', 'field_investigation']
        elif fraud_probability > 0.2:
            return ['enhanced_verification', 'supervisor_review']
        else:
            return ['standard_processing']
    
    def _suggest_verification_steps(self, claim_data):
        """Suggest additional verification steps"""
        steps = []
        
        if claim_data.get('documentation_completeness', 0.9) < 0.8:
            steps.append('request_additional_documentation')
        if claim_data.get('witness_availability', 1) > 0:
            steps.append('interview_witnesses')
        if claim_data.get('medical_provider_risk_score', 3) > 5:
            steps.append('verify_medical_provider_credentials')
        if claim_data.get('injury_severity_vs_incident', 1.0) > 1.5:
            steps.append('independent_medical_examination')
        
        return steps or ['standard_verification_procedures']


def predict_insurance_risk(data):
    """Main function for insurance risk assessment"""
    assessor = InsuranceRiskAssessment()
    return assessor.predict(data)


def predict_insurance_claims(data):
    """Main function for insurance claim prediction"""
    predictor = InsuranceClaimPredictor()
    return predictor.predict(data)


def predict_insurance_fraud(data):
    """Main function for insurance fraud detection"""
    detector = InsuranceFraudDetection()
    return detector.predict(data)