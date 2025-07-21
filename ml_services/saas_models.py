"""
SaaS Industry ML Models
Specialized models for SaaS businesses including churn prediction, usage forecasting, and upsell prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
import logging


class SaaSChurnPredictor:
    """Advanced churn prediction for SaaS businesses"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_saas_data(self, data):
        """SaaS-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # SaaS feature engineering
        features = []
        
        # User engagement metrics
        engagement_features = ['daily_active_usage', 'feature_adoption_score', 'session_duration_avg',
                             'page_views_per_session', 'api_calls_per_day', 'login_frequency',
                             'mobile_app_usage', 'integrations_active', 'custom_fields_used']
        for feature in engagement_features:
            if feature in df.columns:
                features.append(feature)
                
        # Subscription metrics
        subscription_features = ['subscription_length_months', 'plan_tier', 'monthly_recurring_revenue',
                               'payment_method', 'billing_issues_count', 'plan_changes_count',
                               'discount_percentage', 'contract_type', 'renewal_date_days']
        for feature in subscription_features:
            if feature in df.columns:
                features.append(feature)
                
        # Support and satisfaction
        support_features = ['support_tickets_count', 'support_response_time_avg', 'satisfaction_score',
                          'feature_requests_count', 'bug_reports_count', 'training_sessions_attended']
        for feature in support_features:
            if feature in df.columns:
                features.append(feature)
        
        # Usage patterns
        usage_features = ['storage_usage_percentage', 'bandwidth_usage_gb', 'user_seats_utilized',
                        'workflow_automation_usage', 'reporting_usage_frequency', 'export_frequency']
        for feature in usage_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with SaaS-specific logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                if 'usage' in col.lower() or 'active' in col.lower():
                    df[col] = df[col].fillna(0)  # Zero usage indicates potential churn risk
                elif 'score' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 3)
                elif 'frequency' in col.lower():
                    df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 1)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Predict SaaS customer churn"""
        try:
            df, features = self.preprocess_saas_data(data)
            
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
            
            churn_proba = self.model.predict_proba(X_scaled)[0]
            churn_prediction = self.model.predict(X_scaled)[0]
            
            churn_probability = churn_proba[1] if len(churn_proba) > 1 else churn_proba[0]
            
            return {
                'churn_prediction': int(churn_prediction),
                'churn_probability': float(churn_probability),
                'risk_level': self._categorize_saas_risk(churn_probability),
                'confidence': float(max(churn_proba)),
                'engagement_score': self._calculate_engagement_score(df.iloc[0]),
                'risk_factors': self._identify_saas_risk_factors(df.iloc[0]),
                'retention_strategies': self._recommend_saas_retention(churn_probability, df.iloc[0]),
                'feature_adoption_gaps': self._identify_adoption_gaps(df.iloc[0]),
                'customer_health_score': self._calculate_health_score(df.iloc[0]),
                'time_to_churn_estimate': self._estimate_time_to_churn(churn_probability, df.iloc[0])
            }
            
        except Exception as e:
            logging.error(f"SaaS churn prediction error: {str(e)}")
            return {
                'churn_prediction': 0,
                'churn_probability': 0.15,
                'risk_level': 'low',
                'confidence': 0.7,
                'error': f'Prediction error: {str(e)}',
                'risk_factors': ['data_processing_error']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic SaaS data"""
        np.random.seed(42)
        n_samples = 2000
        
        X = np.random.randn(n_samples, len(features))
        
        # SaaS churn logic: low usage, billing issues, poor satisfaction
        y = np.zeros(n_samples)
        for i in range(n_samples):
            churn_score = (X[i, 0] * -0.4 +      # daily usage (low = bad)
                          X[i, 1] * 0.3 +        # billing issues (high = bad)
                          X[i, 2] * -0.3 +       # satisfaction (low = bad)
                          np.random.randn() * 0.1)
            y[i] = 1 if churn_score > 0.2 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_saas_risk(self, probability):
        """Categorize SaaS churn risk"""
        if probability < 0.15:
            return 'low'
        elif probability < 0.35:
            return 'medium'
        elif probability < 0.65:
            return 'high'
        else:
            return 'critical'
    
    def _calculate_engagement_score(self, customer_data):
        """Calculate user engagement score (0-100)"""
        score = 50  # Base score
        
        # Add points for positive engagement
        score += min(25, customer_data.get('daily_active_usage', 1) * 5)
        score += min(15, customer_data.get('feature_adoption_score', 0.5) * 30)
        score += min(10, customer_data.get('integrations_active', 0) * 5)
        
        # Deduct for low engagement
        if customer_data.get('login_frequency', 5) < 2:
            score -= 20
        if customer_data.get('session_duration_avg', 15) < 5:
            score -= 15
        
        return max(0, min(100, int(score)))
    
    def _identify_saas_risk_factors(self, customer_data):
        """Identify SaaS-specific churn risk factors"""
        risks = []
        
        if customer_data.get('daily_active_usage', 5) < 2:
            risks.append('low_daily_usage')
        if customer_data.get('feature_adoption_score', 0.5) < 0.3:
            risks.append('poor_feature_adoption')
        if customer_data.get('billing_issues_count', 0) > 1:
            risks.append('payment_problems')
        if customer_data.get('support_tickets_count', 0) > 5:
            risks.append('high_support_burden')
        if customer_data.get('satisfaction_score', 4) < 3:
            risks.append('low_satisfaction')
        if customer_data.get('login_frequency', 5) < 1:
            risks.append('user_abandonment')
        if customer_data.get('storage_usage_percentage', 50) < 10:
            risks.append('minimal_platform_utilization')
        
        return risks or ['standard_risk_profile']
    
    def _recommend_saas_retention(self, churn_probability, customer_data):
        """SaaS-specific retention strategies"""
        if churn_probability < 0.15:
            return ['feature_expansion_education', 'upsell_opportunities']
        elif churn_probability < 0.35:
            return ['onboarding_improvement', 'feature_training', 'usage_monitoring']
        elif churn_probability < 0.65:
            return ['customer_success_intervention', 'personalized_training', 'discount_offer']
        else:
            return ['urgent_executive_outreach', 'custom_success_plan', 'retention_discount']
    
    def _identify_adoption_gaps(self, customer_data):
        """Identify feature adoption gaps"""
        gaps = []
        
        if customer_data.get('integrations_active', 2) < 1:
            gaps.append('api_integrations')
        if customer_data.get('workflow_automation_usage', 3) < 1:
            gaps.append('automation_features')
        if customer_data.get('reporting_usage_frequency', 2) < 1:
            gaps.append('reporting_analytics')
        if customer_data.get('mobile_app_usage', 5) < 2:
            gaps.append('mobile_platform')
        
        return gaps or ['no_major_gaps']
    
    def _calculate_health_score(self, customer_data):
        """Calculate overall customer health score"""
        engagement = self._calculate_engagement_score(customer_data)
        
        # Adjust for other factors
        health = engagement
        
        if customer_data.get('subscription_length_months', 6) > 12:
            health += 10  # Loyalty bonus
        if customer_data.get('plan_tier', 'basic') in ['premium', 'enterprise']:
            health += 15  # Higher tier bonus
        if customer_data.get('billing_issues_count', 0) > 0:
            health -= 20  # Payment issues penalty
        
        return max(0, min(100, health))
    
    def _estimate_time_to_churn(self, churn_probability, customer_data):
        """Estimate time until potential churn"""
        if churn_probability < 0.15:
            return 'low_risk_next_12_months'
        elif churn_probability < 0.35:
            return '6_to_12_months'
        elif churn_probability < 0.65:
            return '3_to_6_months'
        else:
            return 'within_3_months'


class SaaSUsagePrediction:
    """Predict usage patterns for SaaS platforms"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Predict future usage patterns"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Usage prediction features
            usage_features = ['current_daily_usage', 'user_count', 'storage_used_gb', 'api_calls_monthly',
                            'feature_usage_trend', 'seasonal_factor', 'business_growth_rate',
                            'new_user_onboarding_rate', 'team_size', 'industry_usage_benchmark']
            
            available_features = [f for f in usage_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_usage_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                usage_forecast = self.model.predict(X_scaled)[0]
            else:
                current_usage = df.iloc[0].get('current_daily_usage', 100)
                usage_forecast = current_usage * 1.15  # 15% growth assumption
            
            return {
                'predicted_usage': float(max(0, usage_forecast)),
                'usage_trend': self._determine_usage_trend(df.iloc[0], usage_forecast),
                'capacity_requirements': self._calculate_capacity_needs(usage_forecast),
                'scaling_recommendations': self._recommend_scaling_strategy(usage_forecast, df.iloc[0]),
                'resource_optimization': self._suggest_optimizations(df.iloc[0]),
                'cost_impact': self._estimate_cost_impact(usage_forecast, df.iloc[0]),
                'growth_drivers': self._identify_growth_drivers(df.iloc[0])
            }
            
        except Exception as e:
            return {
                'predicted_usage': 100,
                'usage_trend': 'stable',
                'capacity_requirements': 'current_adequate',
                'error': str(e)
            }
    
    def _train_usage_model(self, features):
        """Train usage prediction model"""
        np.random.seed(42)
        n_samples = 1500
        
        X = np.random.randn(n_samples, len(features))
        
        # Usage growth logic: user growth, feature adoption, business expansion
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_usage = 100
            user_growth = X[i, 0] * 30 if len(features) > 0 else 0
            feature_adoption = X[i, 1] * 20 if len(features) > 1 else 0
            business_growth = X[i, 2] * 25 if len(features) > 2 else 0
            
            y[i] = max(0, base_usage + user_growth + feature_adoption + business_growth + np.random.randn() * 10)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _determine_usage_trend(self, current_data, predicted_usage):
        """Determine usage trend direction"""
        current_usage = current_data.get('current_daily_usage', 100)
        growth_rate = (predicted_usage - current_usage) / current_usage
        
        if growth_rate > 0.2:
            return 'rapid_growth'
        elif growth_rate > 0.05:
            return 'steady_growth'
        elif growth_rate > -0.05:
            return 'stable'
        else:
            return 'declining'
    
    def _calculate_capacity_needs(self, predicted_usage):
        """Calculate infrastructure capacity requirements"""
        if predicted_usage > 1000:
            return 'high_capacity_required'
        elif predicted_usage > 500:
            return 'medium_capacity_adequate'
        else:
            return 'current_capacity_sufficient'
    
    def _recommend_scaling_strategy(self, predicted_usage, current_data):
        """Recommend scaling strategy"""
        current_usage = current_data.get('current_daily_usage', 100)
        
        if predicted_usage > current_usage * 2:
            return 'aggressive_scaling_needed'
        elif predicted_usage > current_usage * 1.5:
            return 'gradual_scaling_recommended'
        else:
            return 'monitor_and_optimize'
    
    def _suggest_optimizations(self, current_data):
        """Suggest resource optimizations"""
        optimizations = []
        
        if current_data.get('storage_used_gb', 50) > 80:
            optimizations.append('implement_data_archiving')
        if current_data.get('api_calls_monthly', 10000) > 50000:
            optimizations.append('api_rate_limiting')
        if current_data.get('user_count', 10) > current_data.get('current_daily_usage', 100):
            optimizations.append('improve_user_engagement')
        
        return optimizations or ['current_setup_optimal']
    
    def _estimate_cost_impact(self, predicted_usage, current_data):
        """Estimate cost implications of usage changes"""
        current_usage = current_data.get('current_daily_usage', 100)
        usage_increase = predicted_usage - current_usage
        
        return {
            'additional_infrastructure_cost': float(usage_increase * 0.1),  # $0.10 per unit
            'support_cost_increase': float(usage_increase * 0.05),
            'total_monthly_impact': float(usage_increase * 0.15 * 30)
        }
    
    def _identify_growth_drivers(self, current_data):
        """Identify factors driving usage growth"""
        drivers = []
        
        if current_data.get('new_user_onboarding_rate', 5) > 10:
            drivers.append('rapid_user_acquisition')
        if current_data.get('feature_usage_trend', 1.0) > 1.2:
            drivers.append('increased_feature_adoption')
        if current_data.get('business_growth_rate', 0.1) > 0.15:
            drivers.append('business_expansion')
        if current_data.get('team_size', 5) > 20:
            drivers.append('enterprise_growth')
        
        return drivers or ['organic_growth']


class SaaSUpsellPredictor:
    """Predict upsell opportunities for SaaS customers"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Predict upsell opportunities"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Upsell prediction features
            upsell_features = ['current_plan_tier', 'usage_vs_limit_ratio', 'feature_limit_hits',
                             'support_tier_requests', 'integration_needs', 'team_growth_rate',
                             'revenue_growth', 'competitive_feature_requests', 'custom_requirements']
            
            available_features = [f for f in upsell_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_upsell_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                upsell_proba = self.model.predict_proba(X_scaled)[0]
                upsell_prediction = self.model.predict(X_scaled)[0]
            else:
                upsell_proba = [0.7, 0.3]
                upsell_prediction = 0
            
            upsell_probability = upsell_proba[1] if len(upsell_proba) > 1 else 0.3
            
            return {
                'upsell_probability': float(upsell_probability),
                'upsell_readiness': self._categorize_upsell_readiness(upsell_probability),
                'recommended_plan': self._recommend_plan_upgrade(df.iloc[0]),
                'value_drivers': self._identify_value_drivers(df.iloc[0]),
                'timing_recommendation': self._recommend_timing(upsell_probability, df.iloc[0]),
                'expected_revenue_increase': self._calculate_revenue_increase(df.iloc[0]),
                'conversion_strategy': self._suggest_conversion_strategy(upsell_probability, df.iloc[0])
            }
            
        except Exception as e:
            return {
                'upsell_probability': 0.3,
                'upsell_readiness': 'moderate',
                'recommended_plan': 'next_tier',
                'error': str(e)
            }
    
    def _train_upsell_model(self, features):
        """Train upsell prediction model"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, len(features))
        
        # Upsell logic: high usage, hitting limits, growth indicators
        y = np.zeros(n_samples)
        for i in range(n_samples):
            upsell_score = (X[i, 0] * 0.4 +      # usage vs limits
                           X[i, 1] * 0.3 +       # feature requests
                           X[i, 2] * 0.2)        # growth indicators
            y[i] = 1 if upsell_score > 0.3 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_upsell_readiness(self, probability):
        """Categorize upsell readiness level"""
        if probability > 0.7:
            return 'high_readiness'
        elif probability > 0.4:
            return 'moderate_readiness'
        else:
            return 'low_readiness'
    
    def _recommend_plan_upgrade(self, customer_data):
        """Recommend specific plan upgrade"""
        current_tier = customer_data.get('current_plan_tier', 'basic')
        usage_ratio = customer_data.get('usage_vs_limit_ratio', 0.5)
        
        if current_tier == 'basic' and usage_ratio > 0.8:
            return 'professional'
        elif current_tier == 'professional' and usage_ratio > 0.9:
            return 'enterprise'
        elif customer_data.get('team_growth_rate', 0.1) > 0.3:
            return 'team_plan'
        else:
            return 'next_tier'
    
    def _identify_value_drivers(self, customer_data):
        """Identify key value drivers for upsell"""
        drivers = []
        
        if customer_data.get('usage_vs_limit_ratio', 0.5) > 0.8:
            drivers.append('capacity_expansion')
        if customer_data.get('feature_limit_hits', 0) > 5:
            drivers.append('feature_unlock')
        if customer_data.get('support_tier_requests', 0) > 3:
            drivers.append('premium_support')
        if customer_data.get('integration_needs', 0) > 2:
            drivers.append('advanced_integrations')
        if customer_data.get('custom_requirements', 0) > 1:
            drivers.append('customization_needs')
        
        return drivers or ['general_upgrade_benefits']
    
    def _recommend_timing(self, upsell_probability, customer_data):
        """Recommend optimal timing for upsell approach"""
        if upsell_probability > 0.7:
            return 'immediate_outreach'
        elif customer_data.get('usage_vs_limit_ratio', 0.5) > 0.9:
            return 'within_next_billing_cycle'
        elif customer_data.get('team_growth_rate', 0.1) > 0.2:
            return 'during_growth_phase'
        else:
            return 'nurture_and_monitor'
    
    def _calculate_revenue_increase(self, customer_data):
        """Calculate expected revenue increase from upsell"""
        current_tier = customer_data.get('current_plan_tier', 'basic')
        
        tier_pricing = {
            'basic': 29,
            'professional': 99,
            'enterprise': 299,
            'team': 149
        }
        
        current_price = tier_pricing.get(current_tier, 29)
        recommended_upgrade = self._recommend_plan_upgrade(customer_data)
        new_price = tier_pricing.get(recommended_upgrade, current_price * 2)
        
        return {
            'monthly_increase': new_price - current_price,
            'annual_increase': (new_price - current_price) * 12
        }
    
    def _suggest_conversion_strategy(self, upsell_probability, customer_data):
        """Suggest conversion strategy"""
        if upsell_probability > 0.7:
            return 'direct_sales_approach'
        elif customer_data.get('usage_vs_limit_ratio', 0.5) > 0.8:
            return 'usage_limit_notification'
        elif customer_data.get('feature_limit_hits', 0) > 3:
            return 'feature_demo_campaign'
        else:
            return 'value_education_nurture'


def predict_saas_churn(data):
    """Main function for SaaS churn prediction"""
    predictor = SaaSChurnPredictor()
    return predictor.predict(data)


def predict_saas_usage(data):
    """Main function for SaaS usage prediction"""
    usage_predictor = SaaSUsagePrediction()
    return usage_predictor.predict(data)


def predict_saas_upsell(data):
    """Main function for SaaS upsell prediction"""
    upsell_predictor = SaaSUpsellPredictor()
    return upsell_predictor.predict(data)