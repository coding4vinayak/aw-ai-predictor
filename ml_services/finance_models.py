"""
Finance Industry ML Models
Specialized models for financial services including credit scoring, fraud detection, and churn prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime
import logging


class FinancialChurnPredictor:
    """Specialized churn prediction for financial institutions"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_financial_data(self, data):
        """Finance-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Financial feature engineering
        features = []
        
        # Account information
        account_features = ['account_age_months', 'account_balance', 'average_balance', 
                          'number_of_products', 'credit_limit', 'credit_utilization']
        for feature in account_features:
            if feature in df.columns:
                features.append(feature)
                
        # Transaction patterns
        transaction_features = ['monthly_transactions', 'atm_usage', 'online_banking_usage',
                              'mobile_app_usage', 'branch_visits', 'international_transactions']
        for feature in transaction_features:
            if feature in df.columns:
                features.append(feature)
                
        # Financial behavior
        behavior_features = ['fee_frequency', 'overdraft_frequency', 'loan_applications',
                           'investment_portfolio_value', 'savings_rate', 'payment_delays']
        for feature in behavior_features:
            if feature in df.columns:
                features.append(feature)
        
        # Customer service interactions
        service_features = ['support_tickets', 'complaint_count', 'satisfaction_score',
                          'response_time_satisfaction', 'product_inquiries']
        for feature in service_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with financial logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                # Use conservative estimates for financial metrics
                if 'balance' in col.lower():
                    df[col] = df[col].fillna(0)
                elif 'score' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 3)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Make financial churn prediction"""
        try:
            df, features = self.preprocess_financial_data(data)
            
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
            
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            # Financial risk factors
            risk_factors = self._identify_financial_risk_factors(df.iloc[0])
            
            return {
                'prediction': int(prediction),
                'churn_probability': float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                'risk_level': self._categorize_financial_risk(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                'confidence': float(max(prediction_proba)),
                'risk_factors': risk_factors,
                'financial_health_score': self._calculate_financial_health(df.iloc[0]),
                'retention_strategies': self._get_retention_strategies(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                'product_recommendations': self._recommend_products(df.iloc[0])
            }
            
        except Exception as e:
            logging.error(f"Financial churn prediction error: {str(e)}")
            return {
                'prediction': 0,
                'churn_probability': 0.25,
                'confidence': 0.6,
                'error': f'Prediction error: {str(e)}',
                'risk_factors': ['data_processing_error']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic financial data"""
        np.random.seed(42)
        n_samples = 1200
        
        X = np.random.randn(n_samples, len(features))
        
        # Financial churn logic: fee sensitivity, low engagement, poor service
        y = np.zeros(n_samples)
        for i in range(n_samples):
            risk_score = (X[i, 0] * 0.4 +     # fee frequency
                         X[i, 1] * -0.3 +     # engagement (negative)
                         X[i, 2] * 0.3)       # complaints
            y[i] = 1 if risk_score > 0.4 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _identify_financial_risk_factors(self, customer_data):
        """Identify financial churn risk factors"""
        risk_factors = []
        
        if customer_data.get('fee_frequency', 0) > 3:
            risk_factors.append('high_fees')
        if customer_data.get('satisfaction_score', 5) < 3:
            risk_factors.append('low_satisfaction')
        if customer_data.get('credit_utilization', 0) > 0.8:
            risk_factors.append('high_credit_utilization')
        if customer_data.get('overdraft_frequency', 0) > 2:
            risk_factors.append('frequent_overdrafts')
        if customer_data.get('complaint_count', 0) > 1:
            risk_factors.append('service_complaints')
        if customer_data.get('mobile_app_usage', 10) < 3:
            risk_factors.append('low_digital_engagement')
        
        return risk_factors or ['standard_risk']
    
    def _categorize_financial_risk(self, probability):
        """Categorize financial churn risk"""
        if probability < 0.2:
            return 'low'
        elif probability < 0.5:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_financial_health(self, customer_data):
        """Calculate overall financial health score"""
        score = 100
        
        # Deduct for negative indicators
        if customer_data.get('overdraft_frequency', 0) > 0:
            score -= customer_data.get('overdraft_frequency', 0) * 10
        if customer_data.get('credit_utilization', 0) > 0.7:
            score -= 20
        if customer_data.get('payment_delays', 0) > 0:
            score -= customer_data.get('payment_delays', 0) * 5
        
        # Add for positive indicators
        if customer_data.get('savings_rate', 0) > 0.1:
            score += 10
        if customer_data.get('investment_portfolio_value', 0) > 10000:
            score += 15
        
        return max(0, min(100, score))
    
    def _get_retention_strategies(self, churn_prob):
        """Financial retention strategies"""
        if churn_prob < 0.2:
            return ['maintain_service_quality', 'cross_sell_opportunities']
        elif churn_prob < 0.5:
            return ['reduce_fees', 'improve_digital_experience', 'loyalty_rewards']
        else:
            return ['immediate_intervention', 'fee_waiver', 'premium_service_upgrade', 'personal_banker_assignment']
    
    def _recommend_products(self, customer_data):
        """Recommend financial products based on profile"""
        recommendations = []
        
        if customer_data.get('savings_rate', 0) > 0.15:
            recommendations.append('high_yield_savings')
        if customer_data.get('investment_portfolio_value', 0) > 50000:
            recommendations.append('premium_investment_advisory')
        if customer_data.get('credit_utilization', 0) < 0.3:
            recommendations.append('credit_limit_increase')
        
        return recommendations or ['basic_checking_optimization']


class CreditScoringModel:
    """Advanced credit scoring for loan applications"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Generate credit score and risk assessment"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Credit scoring features
            credit_features = ['annual_income', 'debt_to_income_ratio', 'credit_history_length',
                             'number_of_credit_accounts', 'payment_history_score', 'credit_utilization',
                             'recent_inquiries', 'derogatory_marks', 'employment_length']
            
            available_features = [f for f in credit_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_credit_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                risk_proba = self.model.predict_proba(X_scaled)[0]
                credit_class = self.model.predict(X_scaled)[0]
            else:
                risk_proba = [0.6, 0.4]
                credit_class = 0
            
            # Convert to credit score scale (300-850)
            credit_score = self._convert_to_credit_score(risk_proba[0] if len(risk_proba) > 1 else 0.6)
            
            return {
                'credit_score': int(credit_score),
                'risk_grade': self._get_risk_grade(credit_score),
                'approval_probability': float(risk_proba[0] if len(risk_proba) > 1 else 0.6),
                'confidence': float(max(risk_proba)),
                'risk_factors': self._identify_credit_risks(df.iloc[0]),
                'improvement_suggestions': self._get_improvement_suggestions(df.iloc[0]),
                'loan_recommendations': self._recommend_loan_products(credit_score)
            }
            
        except Exception as e:
            return {
                'credit_score': 650,
                'risk_grade': 'fair',
                'approval_probability': 0.5,
                'confidence': 0.6,
                'error': str(e)
            }
    
    def _train_credit_model(self, features):
        """Train credit scoring model"""
        np.random.seed(42)
        n_samples = 1500
        
        X = np.random.randn(n_samples, len(features))
        
        # Credit approval logic
        y = np.zeros(n_samples)
        for i in range(n_samples):
            credit_score = (X[i, 0] * 0.35 +      # income
                           X[i, 1] * -0.25 +      # debt ratio (negative)
                           X[i, 2] * 0.2 +        # credit history
                           np.random.randn() * 0.1)
            y[i] = 1 if credit_score > 0.3 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _convert_to_credit_score(self, probability):
        """Convert probability to FICO score scale"""
        # Map probability (0-1) to FICO scale (300-850)
        return 300 + (probability * 550)
    
    def _get_risk_grade(self, credit_score):
        """Get credit risk grade"""
        if credit_score >= 800:
            return 'excellent'
        elif credit_score >= 740:
            return 'very_good'
        elif credit_score >= 670:
            return 'good'
        elif credit_score >= 580:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_credit_risks(self, applicant_data):
        """Identify credit risk factors"""
        risks = []
        
        if applicant_data.get('debt_to_income_ratio', 0) > 0.4:
            risks.append('high_debt_to_income')
        if applicant_data.get('credit_utilization', 0) > 0.7:
            risks.append('high_credit_utilization')
        if applicant_data.get('recent_inquiries', 0) > 5:
            risks.append('multiple_recent_inquiries')
        if applicant_data.get('derogatory_marks', 0) > 0:
            risks.append('negative_credit_history')
        if applicant_data.get('employment_length', 5) < 2:
            risks.append('limited_employment_history')
        
        return risks or ['standard_risk_profile']
    
    def _get_improvement_suggestions(self, applicant_data):
        """Suggest credit improvement strategies"""
        suggestions = []
        
        if applicant_data.get('credit_utilization', 0) > 0.3:
            suggestions.append('reduce_credit_utilization_below_30_percent')
        if applicant_data.get('payment_history_score', 100) < 95:
            suggestions.append('maintain_perfect_payment_history')
        if applicant_data.get('credit_history_length', 5) < 7:
            suggestions.append('maintain_old_accounts_for_longer_history')
        
        return suggestions or ['maintain_current_habits']
    
    def _recommend_loan_products(self, credit_score):
        """Recommend appropriate loan products"""
        if credit_score >= 740:
            return ['prime_mortgage', 'premium_credit_cards', 'personal_loans']
        elif credit_score >= 670:
            return ['conventional_mortgage', 'standard_credit_cards', 'auto_loans']
        elif credit_score >= 580:
            return ['fha_mortgage', 'secured_credit_cards', 'subprime_auto_loans']
        else:
            return ['credit_builder_loans', 'secured_credit_cards', 'financial_counseling']


class FraudDetectionModel:
    """Real-time fraud detection for financial transactions"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Detect fraudulent transaction patterns"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Fraud detection features
            fraud_features = ['transaction_amount', 'time_since_last_transaction', 'merchant_category',
                            'location_distance_from_home', 'transaction_frequency_today',
                            'average_transaction_amount', 'card_usage_pattern_score']
            
            available_features = [f for f in fraud_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_fraud_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                fraud_proba = self.model.predict_proba(X_scaled)[0]
                fraud_prediction = self.model.predict(X_scaled)[0]
            else:
                fraud_proba = [0.95, 0.05]
                fraud_prediction = 0
            
            fraud_score = fraud_proba[1] if len(fraud_proba) > 1 else 0.05
            
            return {
                'fraud_probability': float(fraud_score),
                'fraud_prediction': int(fraud_prediction),
                'risk_level': self._categorize_fraud_risk(fraud_score),
                'confidence': float(max(fraud_proba)),
                'risk_indicators': self._identify_fraud_indicators(df.iloc[0]),
                'recommended_action': self._recommend_fraud_action(fraud_score),
                'monitoring_alerts': self._get_monitoring_alerts(fraud_score)
            }
            
        except Exception as e:
            return {
                'fraud_probability': 0.05,
                'fraud_prediction': 0,
                'risk_level': 'low',
                'confidence': 0.6,
                'error': str(e)
            }
    
    def _train_fraud_model(self, features):
        """Train fraud detection model"""
        np.random.seed(42)
        n_samples = 2000
        
        X = np.random.randn(n_samples, len(features))
        
        # Fraud indicators: unusual amounts, locations, timing
        y = np.zeros(n_samples)
        for i in range(n_samples):
            fraud_score = (abs(X[i, 0]) * 0.4 +      # unusual amount
                          abs(X[i, 1]) * 0.3 +       # location anomaly
                          abs(X[i, 2]) * 0.2)        # timing anomaly
            y[i] = 1 if fraud_score > 1.5 else 0
        
        # Ensure reasonable fraud rate (5%)
        fraud_indices = np.random.choice(np.where(y == 0)[0], size=int(n_samples * 0.05), replace=False)
        y[fraud_indices] = 1
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_fraud_risk(self, probability):
        """Categorize fraud risk level"""
        if probability < 0.1:
            return 'low'
        elif probability < 0.3:
            return 'medium'
        elif probability < 0.7:
            return 'high'
        else:
            return 'critical'
    
    def _identify_fraud_indicators(self, transaction_data):
        """Identify specific fraud indicators"""
        indicators = []
        
        if transaction_data.get('transaction_amount', 0) > 5000:
            indicators.append('high_amount_transaction')
        if transaction_data.get('location_distance_from_home', 0) > 500:
            indicators.append('unusual_location')
        if transaction_data.get('transaction_frequency_today', 1) > 10:
            indicators.append('high_frequency_usage')
        if transaction_data.get('time_since_last_transaction', 60) < 5:
            indicators.append('rapid_successive_transactions')
        
        return indicators or ['normal_transaction_pattern']
    
    def _recommend_fraud_action(self, fraud_score):
        """Recommend action based on fraud score"""
        if fraud_score < 0.1:
            return 'approve_transaction'
        elif fraud_score < 0.3:
            return 'monitor_closely'
        elif fraud_score < 0.7:
            return 'require_additional_verification'
        else:
            return 'block_transaction_immediately'
    
    def _get_monitoring_alerts(self, fraud_score):
        """Generate monitoring alerts"""
        alerts = []
        
        if fraud_score > 0.3:
            alerts.append('enhanced_monitoring_activated')
        if fraud_score > 0.5:
            alerts.append('manual_review_required')
        if fraud_score > 0.7:
            alerts.append('immediate_security_team_notification')
        
        return alerts


def predict_financial_churn(data):
    """Main function for financial churn prediction"""
    predictor = FinancialChurnPredictor()
    return predictor.predict(data)


def predict_credit_score(data):
    """Main function for credit scoring"""
    scorer = CreditScoringModel()
    return scorer.predict(data)


def predict_fraud_detection(data):
    """Main function for fraud detection"""
    detector = FraudDetectionModel()
    return detector.predict(data)