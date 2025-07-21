"""
Retail Industry ML Models
Specialized models for retail businesses including demand forecasting, price optimization, and customer analytics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
import logging


class RetailDemandForecaster:
    """Advanced demand forecasting for retail inventory management"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_retail_data(self, data):
        """Retail-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Retail feature engineering
        features = []
        
        # Product characteristics
        product_features = ['category', 'subcategory', 'brand', 'price', 'cost', 'margin',
                          'seasonality_index', 'promotion_active', 'competitor_price']
        for feature in product_features:
            if feature in df.columns:
                features.append(feature)
                
        # Historical sales patterns
        sales_features = ['sales_last_week', 'sales_last_month', 'sales_last_year',
                        'inventory_level', 'stock_turnover', 'days_of_supply']
        for feature in sales_features:
            if feature in df.columns:
                features.append(feature)
                
        # External factors
        external_features = ['weather_impact', 'holiday_indicator', 'economic_index',
                           'marketing_spend', 'social_media_mentions', 'competitor_actions']
        for feature in external_features:
            if feature in df.columns:
                features.append(feature)
        
        # Store characteristics
        store_features = ['store_size', 'location_type', 'foot_traffic', 'demographic_score']
        for feature in store_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with retail-specific logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                if 'sales' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
                elif 'price' in col.lower():
                    df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 10)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Generate demand forecast"""
        try:
            df, features = self.preprocess_retail_data(data)
            
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
            demand_forecast = self.model.predict(X_scaled)[0]
            
            # Generate forecast confidence intervals
            forecast_std = self._estimate_forecast_uncertainty(X_scaled)
            
            # Retail-specific insights
            insights = self._generate_retail_insights(df.iloc[0], demand_forecast)
            
            return {
                'demand_forecast': float(max(0, demand_forecast)),
                'forecast_period': 'next_30_days',
                'confidence_interval': {
                    'lower': float(max(0, demand_forecast - 1.96 * forecast_std)),
                    'upper': float(demand_forecast + 1.96 * forecast_std)
                },
                'seasonality_impact': self._assess_seasonality_impact(df.iloc[0]),
                'key_drivers': self._identify_demand_drivers(df.iloc[0]),
                'inventory_recommendations': self._recommend_inventory_actions(demand_forecast, df.iloc[0]),
                'pricing_insights': insights['pricing'],
                'marketing_recommendations': insights['marketing']
            }
            
        except Exception as e:
            logging.error(f"Retail demand forecast error: {str(e)}")
            return {
                'demand_forecast': 100,
                'forecast_period': 'next_30_days',
                'confidence_interval': {'lower': 80, 'upper': 120},
                'error': f'Forecast error: {str(e)}',
                'key_drivers': ['historical_trends']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic retail data"""
        np.random.seed(42)
        n_samples = 1500
        
        X = np.random.randn(n_samples, len(features))
        
        # Retail demand logic: seasonality, pricing, promotions
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_demand = 100
            seasonal_effect = X[i, 0] * 20 if len(features) > 0 else 0
            price_effect = -X[i, 1] * 15 if len(features) > 1 else 0
            promotion_effect = X[i, 2] * 25 if len(features) > 2 else 0
            
            y[i] = max(0, base_demand + seasonal_effect + price_effect + promotion_effect + np.random.randn() * 10)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _estimate_forecast_uncertainty(self, X_scaled):
        """Estimate forecast uncertainty"""
        # Simple uncertainty estimation
        return 15.0  # 15% standard deviation
    
    def _assess_seasonality_impact(self, product_data):
        """Assess seasonal impact on demand"""
        seasonality = product_data.get('seasonality_index', 1.0)
        
        if seasonality > 1.2:
            return 'high_seasonal_boost'
        elif seasonality < 0.8:
            return 'seasonal_decline'
        else:
            return 'neutral_seasonal_impact'
    
    def _identify_demand_drivers(self, product_data):
        """Identify key demand drivers"""
        drivers = []
        
        if product_data.get('promotion_active', 0) == 1:
            drivers.append('active_promotion')
        if product_data.get('competitor_price', 100) > product_data.get('price', 90):
            drivers.append('competitive_pricing')
        if product_data.get('marketing_spend', 0) > 1000:
            drivers.append('marketing_campaign')
        if product_data.get('weather_impact', 0) > 0.5:
            drivers.append('weather_conditions')
        
        return drivers or ['base_demand_trends']
    
    def _recommend_inventory_actions(self, forecast, product_data):
        """Recommend inventory management actions"""
        current_inventory = product_data.get('inventory_level', 100)
        
        if forecast > current_inventory * 1.5:
            return 'increase_inventory_order'
        elif forecast < current_inventory * 0.5:
            return 'reduce_inventory_order'
        else:
            return 'maintain_current_levels'
    
    def _generate_retail_insights(self, product_data, forecast):
        """Generate comprehensive retail insights"""
        pricing_insight = 'optimal_price_range'
        marketing_insight = 'maintain_current_strategy'
        
        if product_data.get('margin', 0.3) < 0.2:
            pricing_insight = 'consider_price_increase'
        elif product_data.get('competitor_price', 100) < product_data.get('price', 90) * 0.9:
            pricing_insight = 'competitive_pressure_detected'
        
        if forecast > product_data.get('sales_last_month', 80) * 1.3:
            marketing_insight = 'increase_marketing_investment'
        
        return {
            'pricing': pricing_insight,
            'marketing': marketing_insight
        }


class RetailPriceOptimizer:
    """Price optimization for maximum profitability"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Optimize pricing strategy"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Price optimization features
            price_features = ['current_price', 'cost', 'competitor_price', 'demand_elasticity',
                            'brand_premium', 'market_position', 'inventory_level', 'seasonality']
            
            available_features = [f for f in price_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_price_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                optimal_price = self.model.predict(X_scaled)[0]
            else:
                optimal_price = data.get('current_price', 50) * 1.05
            
            current_price = df.iloc[0].get('current_price', 50)
            
            return {
                'optimal_price': float(max(0, optimal_price)),
                'current_price': float(current_price),
                'price_change_recommendation': self._get_price_change_recommendation(current_price, optimal_price),
                'expected_impact': self._calculate_price_impact(current_price, optimal_price, df.iloc[0]),
                'risk_assessment': self._assess_pricing_risk(current_price, optimal_price, df.iloc[0]),
                'competitor_analysis': self._analyze_competitive_position(df.iloc[0]),
                'implementation_strategy': self._recommend_implementation_strategy(current_price, optimal_price)
            }
            
        except Exception as e:
            return {
                'optimal_price': data.get('current_price', 50),
                'price_change_recommendation': 'maintain_current_price',
                'error': str(e)
            }
    
    def _train_price_model(self, features):
        """Train price optimization model"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, len(features))
        
        # Price optimization logic: cost plus margin, competition, demand
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_price = 50
            cost_factor = X[i, 0] * 10 if len(features) > 0 else 0
            competition_factor = X[i, 1] * 5 if len(features) > 1 else 0
            demand_factor = X[i, 2] * 8 if len(features) > 2 else 0
            
            y[i] = max(10, base_price + cost_factor + competition_factor + demand_factor)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _get_price_change_recommendation(self, current_price, optimal_price):
        """Recommend price change action"""
        price_diff = (optimal_price - current_price) / current_price
        
        if abs(price_diff) < 0.02:
            return 'maintain_current_price'
        elif price_diff > 0.1:
            return 'gradual_price_increase'
        elif price_diff > 0.05:
            return 'moderate_price_increase'
        elif price_diff < -0.1:
            return 'significant_price_reduction'
        else:
            return 'minor_price_adjustment'
    
    def _calculate_price_impact(self, current_price, optimal_price, product_data):
        """Calculate expected impact of price change"""
        price_change = (optimal_price - current_price) / current_price
        demand_elasticity = product_data.get('demand_elasticity', -1.2)
        
        demand_change = price_change * demand_elasticity
        revenue_change = price_change + demand_change + (price_change * demand_change)
        
        return {
            'expected_demand_change': f"{demand_change:.2%}",
            'expected_revenue_change': f"{revenue_change:.2%}",
            'profit_margin_impact': self._calculate_margin_impact(current_price, optimal_price, product_data)
        }
    
    def _calculate_margin_impact(self, current_price, optimal_price, product_data):
        """Calculate impact on profit margin"""
        cost = product_data.get('cost', current_price * 0.7)
        
        current_margin = (current_price - cost) / current_price
        new_margin = (optimal_price - cost) / optimal_price
        
        return f"{(new_margin - current_margin):.2%}"
    
    def _assess_pricing_risk(self, current_price, optimal_price, product_data):
        """Assess risks of price change"""
        price_change = abs(optimal_price - current_price) / current_price
        competitor_price = product_data.get('competitor_price', current_price)
        
        risks = []
        
        if price_change > 0.15:
            risks.append('large_price_change_risk')
        if optimal_price > competitor_price * 1.1:
            risks.append('competitive_disadvantage_risk')
        if product_data.get('inventory_level', 100) > 500:
            risks.append('inventory_clearance_pressure')
        
        return risks or ['low_risk']
    
    def _analyze_competitive_position(self, product_data):
        """Analyze competitive pricing position"""
        current_price = product_data.get('current_price', 50)
        competitor_price = product_data.get('competitor_price', current_price)
        
        if current_price < competitor_price * 0.9:
            return 'price_leader_position'
        elif current_price > competitor_price * 1.1:
            return 'premium_position'
        else:
            return 'competitive_parity'
    
    def _recommend_implementation_strategy(self, current_price, optimal_price):
        """Recommend how to implement price change"""
        price_diff = abs(optimal_price - current_price) / current_price
        
        if price_diff < 0.05:
            return 'immediate_implementation'
        elif price_diff < 0.15:
            return 'two_phase_implementation'
        else:
            return 'gradual_phased_approach'


class RetailChurnPredictor:
    """Customer churn prediction for retail businesses"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Predict customer churn probability"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Retail churn features
            churn_features = ['recency_days', 'frequency_purchases', 'monetary_value', 'avg_order_value',
                            'category_diversity', 'discount_usage', 'return_rate', 'support_tickets',
                            'app_usage_frequency', 'email_engagement', 'loyalty_points_balance']
            
            available_features = [f for f in churn_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_churn_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                churn_proba = self.model.predict_proba(X_scaled)[0]
                churn_prediction = self.model.predict(X_scaled)[0]
            else:
                churn_proba = [0.75, 0.25]
                churn_prediction = 0
            
            churn_probability = churn_proba[1] if len(churn_proba) > 1 else 0.25
            
            return {
                'churn_probability': float(churn_probability),
                'churn_prediction': int(churn_prediction),
                'customer_segment': self._segment_customer(df.iloc[0]),
                'risk_factors': self._identify_churn_risks(df.iloc[0]),
                'retention_strategies': self._recommend_retention_strategies(churn_probability, df.iloc[0]),
                'clv_impact': self._calculate_clv_impact(df.iloc[0], churn_probability),
                'next_best_actions': self._suggest_next_actions(churn_probability, df.iloc[0])
            }
            
        except Exception as e:
            return {
                'churn_probability': 0.25,
                'churn_prediction': 0,
                'customer_segment': 'standard',
                'error': str(e)
            }
    
    def _train_churn_model(self, features):
        """Train retail churn model"""
        np.random.seed(42)
        n_samples = 1200
        
        X = np.random.randn(n_samples, len(features))
        
        # Retail churn logic: recency, frequency, monetary value
        y = np.zeros(n_samples)
        for i in range(n_samples):
            rfm_score = (X[i, 0] * 0.4 +      # recency (high = bad)
                        X[i, 1] * -0.3 +      # frequency (high = good)
                        X[i, 2] * -0.2)       # monetary (high = good)
            y[i] = 1 if rfm_score > 0.3 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _segment_customer(self, customer_data):
        """Segment customer based on RFM analysis"""
        recency = customer_data.get('recency_days', 30)
        frequency = customer_data.get('frequency_purchases', 5)
        monetary = customer_data.get('monetary_value', 500)
        
        if recency <= 30 and frequency >= 10 and monetary >= 1000:
            return 'champion'
        elif recency <= 60 and frequency >= 5 and monetary >= 500:
            return 'loyal_customer'
        elif recency > 90 and frequency < 3:
            return 'at_risk'
        elif recency > 180:
            return 'hibernating'
        else:
            return 'potential_loyalist'
    
    def _identify_churn_risks(self, customer_data):
        """Identify specific churn risk factors"""
        risks = []
        
        if customer_data.get('recency_days', 30) > 60:
            risks.append('long_time_since_purchase')
        if customer_data.get('frequency_purchases', 5) < 2:
            risks.append('low_purchase_frequency')
        if customer_data.get('return_rate', 0.1) > 0.2:
            risks.append('high_return_rate')
        if customer_data.get('support_tickets', 0) > 3:
            risks.append('service_issues')
        if customer_data.get('email_engagement', 0.5) < 0.2:
            risks.append('low_engagement')
        
        return risks or ['standard_risk_profile']
    
    def _recommend_retention_strategies(self, churn_probability, customer_data):
        """Recommend retention strategies"""
        if churn_probability < 0.3:
            return ['maintain_engagement', 'loyalty_program_enrollment']
        elif churn_probability < 0.6:
            return ['personalized_offers', 'win_back_campaign', 'customer_service_outreach']
        else:
            return ['urgent_intervention', 'deep_discount_offer', 'personal_shopping_service']
    
    def _calculate_clv_impact(self, customer_data, churn_probability):
        """Calculate customer lifetime value impact"""
        avg_order_value = customer_data.get('avg_order_value', 100)
        frequency = customer_data.get('frequency_purchases', 5)
        
        annual_value = avg_order_value * frequency
        clv_at_risk = annual_value * 3 * churn_probability  # 3-year horizon
        
        return {
            'potential_lost_clv': float(clv_at_risk),
            'retention_roi': float(clv_at_risk * 0.1)  # 10% intervention cost
        }
    
    def _suggest_next_actions(self, churn_probability, customer_data):
        """Suggest immediate next actions"""
        if churn_probability > 0.7:
            return ['immediate_phone_call', 'exclusive_vip_offer']
        elif churn_probability > 0.4:
            return ['personalized_email_campaign', 'product_recommendations']
        else:
            return ['newsletter_optimization', 'cross_sell_opportunities']


def predict_retail_demand(data):
    """Main function for retail demand forecasting"""
    forecaster = RetailDemandForecaster()
    return forecaster.predict(data)


def predict_retail_price(data):
    """Main function for retail price optimization"""
    optimizer = RetailPriceOptimizer()
    return optimizer.predict(data)


def predict_retail_churn(data):
    """Main function for retail churn prediction"""
    predictor = RetailChurnPredictor()
    return predictor.predict(data)