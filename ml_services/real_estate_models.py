"""
Real Estate Industry ML Models
Specialized models for real estate including price prediction, market analysis, and investment scoring
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
import logging


class RealEstatePricePredictor:
    """Advanced property price prediction and valuation"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=12,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_property_data(self, data):
        """Real estate specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Real estate feature engineering
        features = []
        
        # Property characteristics
        property_features = ['square_footage', 'bedrooms', 'bathrooms', 'lot_size', 'year_built',
                           'garage_spaces', 'stories', 'basement_sqft', 'property_type', 'architectural_style']
        for feature in property_features:
            if feature in df.columns:
                features.append(feature)
                
        # Location factors
        location_features = ['zip_code', 'neighborhood_score', 'school_district_rating', 'crime_rate_index',
                           'walkability_score', 'public_transport_access', 'distance_to_downtown',
                           'nearby_amenities_count', 'park_proximity_score']
        for feature in location_features:
            if feature in df.columns:
                features.append(feature)
                
        # Market conditions
        market_features = ['days_on_market', 'list_price', 'price_per_sqft_area', 'market_trend_index',
                         'inventory_levels', 'median_income_area', 'unemployment_rate_area',
                         'population_growth_rate', 'new_construction_rate']
        for feature in market_features:
            if feature in df.columns:
                features.append(feature)
        
        # Property condition and features
        condition_features = ['condition_score', 'renovation_year', 'hvac_age', 'roof_age', 'appliances_included',
                            'luxury_features_count', 'energy_efficiency_rating', 'smart_home_features']
        for feature in condition_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with real estate logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                if 'score' in col.lower() or 'rating' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 5)
                elif 'price' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 250000)
                elif 'age' in col.lower() or 'year' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 20)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Predict property value and market insights"""
        try:
            df, features = self.preprocess_property_data(data)
            
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
            predicted_price = self.model.predict(X_scaled)[0]
            
            # Ensure reasonable price range
            predicted_price = max(50000, predicted_price)
            
            return {
                'predicted_price': float(predicted_price),
                'price_range': self._calculate_price_range(predicted_price),
                'price_per_sqft': self._calculate_price_per_sqft(predicted_price, df.iloc[0]),
                'market_position': self._analyze_market_position(predicted_price, df.iloc[0]),
                'value_drivers': self._identify_value_drivers(df.iloc[0]),
                'pricing_recommendation': self._recommend_pricing_strategy(predicted_price, df.iloc[0]),
                'investment_metrics': self._calculate_investment_metrics(predicted_price, df.iloc[0]),
                'comparable_analysis': self._generate_comparable_insights(predicted_price, df.iloc[0])
            }
            
        except Exception as e:
            logging.error(f"Real estate price prediction error: {str(e)}")
            return {
                'predicted_price': 250000,
                'price_range': {'low': 225000, 'high': 275000},
                'price_per_sqft': 150,
                'error': f'Prediction error: {str(e)}',
                'value_drivers': ['location', 'size']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic real estate data"""
        np.random.seed(42)
        n_samples = 2500
        
        X = np.random.randn(n_samples, len(features))
        
        # Price logic: size, location, condition, market factors
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_price = 200000
            size_factor = X[i, 0] * 50000 if len(features) > 0 else 0      # Square footage impact
            location_factor = X[i, 1] * 80000 if len(features) > 1 else 0  # Location premium
            condition_factor = X[i, 2] * 30000 if len(features) > 2 else 0 # Condition impact
            market_factor = X[i, 3] * 40000 if len(features) > 3 else 0    # Market conditions
            
            y[i] = max(50000, base_price + size_factor + location_factor + condition_factor + 
                      market_factor + np.random.randn() * 15000)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _calculate_price_range(self, predicted_price):
        """Calculate confidence interval for price prediction"""
        margin = predicted_price * 0.1  # 10% margin
        return {
            'low': float(predicted_price - margin),
            'high': float(predicted_price + margin)
        }
    
    def _calculate_price_per_sqft(self, predicted_price, property_data):
        """Calculate price per square foot"""
        sqft = property_data.get('square_footage', 1500)
        return float(predicted_price / sqft) if sqft > 0 else 150
    
    def _analyze_market_position(self, predicted_price, property_data):
        """Analyze property's position in the market"""
        area_median = property_data.get('price_per_sqft_area', 150) * property_data.get('square_footage', 1500)
        
        if predicted_price > area_median * 1.2:
            return 'premium_segment'
        elif predicted_price > area_median * 0.9:
            return 'market_rate'
        else:
            return 'value_segment'
    
    def _identify_value_drivers(self, property_data):
        """Identify key factors driving property value"""
        drivers = []
        
        if property_data.get('school_district_rating', 5) > 8:
            drivers.append('excellent_schools')
        if property_data.get('square_footage', 1500) > 2500:
            drivers.append('large_size')
        if property_data.get('neighborhood_score', 5) > 8:
            drivers.append('desirable_location')
        if property_data.get('luxury_features_count', 0) > 3:
            drivers.append('luxury_amenities')
        if property_data.get('walkability_score', 5) > 8:
            drivers.append('walkable_neighborhood')
        if property_data.get('year_built', 1980) > 2010:
            drivers.append('modern_construction')
        
        return drivers or ['standard_property_features']
    
    def _recommend_pricing_strategy(self, predicted_price, property_data):
        """Recommend pricing strategy for listing"""
        list_price = property_data.get('list_price', predicted_price)
        
        if list_price > predicted_price * 1.05:
            return 'reduce_list_price'
        elif list_price < predicted_price * 0.95:
            return 'increase_list_price'
        else:
            return 'competitively_priced'
    
    def _calculate_investment_metrics(self, predicted_price, property_data):
        """Calculate investment-related metrics"""
        potential_rent = predicted_price * 0.005  # 0.5% of value as monthly rent estimate
        
        return {
            'estimated_monthly_rent': float(potential_rent),
            'gross_yield_estimate': float((potential_rent * 12 / predicted_price) * 100),
            'appreciation_potential': self._assess_appreciation_potential(property_data),
            'investment_grade': self._grade_investment_potential(predicted_price, property_data)
        }
    
    def _assess_appreciation_potential(self, property_data):
        """Assess property appreciation potential"""
        growth_rate = property_data.get('population_growth_rate', 0.02)
        market_trend = property_data.get('market_trend_index', 1.0)
        
        if growth_rate > 0.05 and market_trend > 1.1:
            return 'high'
        elif growth_rate > 0.02 or market_trend > 1.05:
            return 'moderate'
        else:
            return 'low'
    
    def _grade_investment_potential(self, predicted_price, property_data):
        """Grade overall investment potential"""
        factors = 0
        
        if property_data.get('neighborhood_score', 5) > 7:
            factors += 1
        if property_data.get('school_district_rating', 5) > 7:
            factors += 1
        if property_data.get('population_growth_rate', 0.02) > 0.03:
            factors += 1
        if property_data.get('crime_rate_index', 5) < 3:
            factors += 1
        
        if factors >= 3:
            return 'A'
        elif factors >= 2:
            return 'B'
        else:
            return 'C'
    
    def _generate_comparable_insights(self, predicted_price, property_data):
        """Generate insights about comparable properties"""
        return {
            'estimated_comparable_count': 15,
            'price_variance_in_area': '±12%',
            'days_on_market_expected': self._estimate_days_on_market(predicted_price, property_data),
            'competitive_advantages': self._identify_competitive_advantages(property_data)
        }
    
    def _estimate_days_on_market(self, predicted_price, property_data):
        """Estimate expected days on market"""
        list_price = property_data.get('list_price', predicted_price)
        price_ratio = list_price / predicted_price
        
        if price_ratio <= 1.0:
            return 30
        elif price_ratio <= 1.1:
            return 45
        else:
            return 75
    
    def _identify_competitive_advantages(self, property_data):
        """Identify competitive advantages"""
        advantages = []
        
        if property_data.get('energy_efficiency_rating', 5) > 8:
            advantages.append('energy_efficient')
        if property_data.get('smart_home_features', 0) > 3:
            advantages.append('smart_home_technology')
        if property_data.get('renovation_year', 2000) > 2018:
            advantages.append('recently_renovated')
        if property_data.get('garage_spaces', 1) > 2:
            advantages.append('ample_parking')
        
        return advantages or ['standard_features']


class RealEstateMarketAnalyzer:
    """Analyze real estate market trends and conditions"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Analyze market conditions and trends"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Market analysis features
            market_features = ['median_home_price', 'inventory_months', 'sales_volume_change', 'price_growth_rate',
                             'new_listings_rate', 'pending_sales_ratio', 'days_on_market_avg', 'price_reductions_rate',
                             'mortgage_rates', 'economic_index', 'population_growth', 'employment_rate']
            
            available_features = [f for f in market_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_market_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                market_score = self.model.predict(X_scaled)[0]
            else:
                market_score = 50  # Neutral market
            
            return {
                'market_health_score': float(max(0, min(100, market_score))),
                'market_condition': self._categorize_market_condition(market_score),
                'trend_direction': self._determine_trend_direction(df.iloc[0]),
                'buyer_seller_advantage': self._assess_market_balance(df.iloc[0]),
                'investment_timing': self._recommend_investment_timing(market_score, df.iloc[0]),
                'market_risks': self._identify_market_risks(df.iloc[0]),
                'opportunity_score': self._calculate_opportunity_score(market_score, df.iloc[0]),
                'forecast_indicators': self._analyze_forecast_indicators(df.iloc[0])
            }
            
        except Exception as e:
            return {
                'market_health_score': 50,
                'market_condition': 'balanced',
                'trend_direction': 'stable',
                'error': str(e)
            }
    
    def _train_market_model(self, features):
        """Train market analysis model"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, len(features))
        
        # Market health logic: inventory, sales, price trends
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_health = 50
            inventory_factor = X[i, 0] * -10 if len(features) > 0 else 0    # High inventory = lower health
            sales_factor = X[i, 1] * 15 if len(features) > 1 else 0         # High sales = better health
            price_factor = X[i, 2] * 10 if len(features) > 2 else 0         # Price growth = health
            
            y[i] = max(0, min(100, base_health + inventory_factor + sales_factor + price_factor + np.random.randn() * 5))
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_market_condition(self, score):
        """Categorize overall market condition"""
        if score >= 75:
            return 'strong_sellers_market'
        elif score >= 60:
            return 'sellers_market'
        elif score >= 40:
            return 'balanced_market'
        elif score >= 25:
            return 'buyers_market'
        else:
            return 'strong_buyers_market'
    
    def _determine_trend_direction(self, market_data):
        """Determine market trend direction"""
        price_growth = market_data.get('price_growth_rate', 0.05)
        sales_change = market_data.get('sales_volume_change', 0.0)
        
        if price_growth > 0.08 and sales_change > 0.1:
            return 'rapidly_appreciating'
        elif price_growth > 0.03 or sales_change > 0.05:
            return 'appreciating'
        elif price_growth > -0.03 and sales_change > -0.05:
            return 'stable'
        else:
            return 'declining'
    
    def _assess_market_balance(self, market_data):
        """Assess whether market favors buyers or sellers"""
        inventory_months = market_data.get('inventory_months', 6)
        
        if inventory_months < 3:
            return 'strong_sellers_advantage'
        elif inventory_months < 5:
            return 'sellers_advantage'
        elif inventory_months < 7:
            return 'balanced'
        elif inventory_months < 10:
            return 'buyers_advantage'
        else:
            return 'strong_buyers_advantage'
    
    def _recommend_investment_timing(self, market_score, market_data):
        """Recommend timing for real estate investment"""
        if market_score > 70:
            return 'consider_waiting_for_correction'
        elif market_score > 50:
            return 'good_time_for_quality_properties'
        elif market_score > 30:
            return 'excellent_buying_opportunity'
        else:
            return 'wait_for_market_stabilization'
    
    def _identify_market_risks(self, market_data):
        """Identify potential market risks"""
        risks = []
        
        if market_data.get('inventory_months', 6) < 2:
            risks.append('overheated_market')
        if market_data.get('price_growth_rate', 0.05) > 0.15:
            risks.append('unsustainable_price_growth')
        if market_data.get('mortgage_rates', 0.04) > 0.07:
            risks.append('high_borrowing_costs')
        if market_data.get('employment_rate', 0.95) < 0.92:
            risks.append('economic_uncertainty')
        if market_data.get('new_listings_rate', 1.0) > 1.5:
            risks.append('increasing_supply')
        
        return risks or ['normal_market_risks']
    
    def _calculate_opportunity_score(self, market_score, market_data):
        """Calculate investment opportunity score"""
        base_score = market_score
        
        # Adjust for specific opportunities
        if market_data.get('price_reductions_rate', 0.1) > 0.2:
            base_score += 10  # Many price reductions = opportunity
        if market_data.get('days_on_market_avg', 30) > 60:
            base_score += 5   # Longer DOM = negotiation opportunity
        
        return min(100, max(0, base_score))
    
    def _analyze_forecast_indicators(self, market_data):
        """Analyze indicators for future market direction"""
        indicators = []
        
        if market_data.get('pending_sales_ratio', 0.8) > 0.9:
            indicators.append('strong_future_sales_pipeline')
        if market_data.get('new_listings_rate', 1.0) < 0.8:
            indicators.append('supply_constraints_continuing')
        if market_data.get('mortgage_rates', 0.04) < 0.05:
            indicators.append('favorable_financing_conditions')
        if market_data.get('population_growth', 0.02) > 0.03:
            indicators.append('increasing_demand_pressure')
        
        return indicators or ['mixed_signals']


class RealEstateInvestmentScorer:
    """Score real estate investment opportunities"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Score real estate investment opportunity"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Investment scoring features
            investment_features = ['cap_rate', 'cash_flow_monthly', 'appreciation_potential', 'rental_yield',
                                 'vacancy_rate_area', 'property_condition_score', 'location_growth_score',
                                 'financing_terms_score', 'exit_strategy_options', 'market_liquidity_score']
            
            available_features = [f for f in investment_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_investment_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                investment_proba = self.model.predict_proba(X_scaled)[0]
                investment_prediction = self.model.predict(X_scaled)[0]
            else:
                investment_proba = [0.4, 0.6]
                investment_prediction = 1
            
            investment_score = investment_proba[1] if len(investment_proba) > 1 else 0.6
            
            return {
                'investment_score': float(investment_score * 100),
                'investment_grade': self._grade_investment(investment_score),
                'risk_return_profile': self._assess_risk_return(df.iloc[0]),
                'cash_flow_analysis': self._analyze_cash_flow(df.iloc[0]),
                'appreciation_forecast': self._forecast_appreciation(df.iloc[0]),
                'investment_strengths': self._identify_strengths(df.iloc[0]),
                'investment_concerns': self._identify_concerns(df.iloc[0]),
                'recommended_strategy': self._recommend_strategy(investment_score, df.iloc[0])
            }
            
        except Exception as e:
            return {
                'investment_score': 60,
                'investment_grade': 'B',
                'risk_return_profile': 'moderate_risk_moderate_return',
                'error': str(e)
            }
    
    def _train_investment_model(self, features):
        """Train investment scoring model"""
        np.random.seed(42)
        n_samples = 1200
        
        X = np.random.randn(n_samples, len(features))
        
        # Investment quality logic: returns, cash flow, appreciation, risk
        y = np.zeros(n_samples)
        for i in range(n_samples):
            quality_score = (X[i, 0] * 0.3 +      # cap rate/returns
                           X[i, 1] * 0.3 +        # cash flow
                           X[i, 2] * 0.2 +        # appreciation potential
                           X[i, 3] * -0.1)        # risk factors (negative)
            y[i] = 1 if quality_score > 0.1 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _grade_investment(self, score):
        """Grade investment opportunity"""
        if score >= 0.8:
            return 'A+'
        elif score >= 0.7:
            return 'A'
        elif score >= 0.6:
            return 'B+'
        elif score >= 0.5:
            return 'B'
        elif score >= 0.4:
            return 'C+'
        else:
            return 'C'
    
    def _assess_risk_return(self, investment_data):
        """Assess risk-return profile"""
        cap_rate = investment_data.get('cap_rate', 0.06)
        vacancy_rate = investment_data.get('vacancy_rate_area', 0.05)
        
        if cap_rate > 0.08 and vacancy_rate < 0.05:
            return 'high_return_low_risk'
        elif cap_rate > 0.08:
            return 'high_return_high_risk'
        elif vacancy_rate < 0.05:
            return 'moderate_return_low_risk'
        else:
            return 'moderate_return_moderate_risk'
    
    def _analyze_cash_flow(self, investment_data):
        """Analyze cash flow characteristics"""
        monthly_cf = investment_data.get('cash_flow_monthly', 500)
        
        return {
            'monthly_cash_flow': float(monthly_cf),
            'annual_cash_flow': float(monthly_cf * 12),
            'cash_flow_stability': 'stable' if monthly_cf > 0 else 'negative',
            'break_even_analysis': 'positive' if monthly_cf > 0 else 'requires_capital'
        }
    
    def _forecast_appreciation(self, investment_data):
        """Forecast property appreciation"""
        appreciation_potential = investment_data.get('appreciation_potential', 0.03)
        
        if appreciation_potential > 0.06:
            return 'strong_appreciation_expected'
        elif appreciation_potential > 0.03:
            return 'moderate_appreciation_likely'
        else:
            return 'limited_appreciation_potential'
    
    def _identify_strengths(self, investment_data):
        """Identify investment strengths"""
        strengths = []
        
        if investment_data.get('cap_rate', 0.06) > 0.08:
            strengths.append('high_cap_rate')
        if investment_data.get('cash_flow_monthly', 0) > 1000:
            strengths.append('strong_cash_flow')
        if investment_data.get('location_growth_score', 5) > 8:
            strengths.append('excellent_location')
        if investment_data.get('property_condition_score', 5) > 8:
            strengths.append('excellent_condition')
        if investment_data.get('market_liquidity_score', 5) > 7:
            strengths.append('high_liquidity')
        
        return strengths or ['standard_investment_profile']
    
    def _identify_concerns(self, investment_data):
        """Identify investment concerns"""
        concerns = []
        
        if investment_data.get('vacancy_rate_area', 0.05) > 0.1:
            concerns.append('high_vacancy_risk')
        if investment_data.get('cash_flow_monthly', 500) < 0:
            concerns.append('negative_cash_flow')
        if investment_data.get('property_condition_score', 8) < 5:
            concerns.append('property_condition_issues')
        if investment_data.get('financing_terms_score', 7) < 5:
            concerns.append('challenging_financing')
        if investment_data.get('exit_strategy_options', 3) < 2:
            concerns.append('limited_exit_options')
        
        return concerns or ['minimal_concerns']
    
    def _recommend_strategy(self, investment_score, investment_data):
        """Recommend investment strategy"""
        if investment_score > 0.8:
            return 'strong_buy_recommendation'
        elif investment_score > 0.6:
            return 'buy_with_standard_due_diligence'
        elif investment_score > 0.4:
            return 'conditional_buy_address_concerns'
        else:
            return 'pass_seek_better_opportunities'


def predict_real_estate_price(data):
    """Main function for real estate price prediction"""
    predictor = RealEstatePricePredictor()
    return predictor.predict(data)


def predict_real_estate_market(data):
    """Main function for real estate market analysis"""
    analyzer = RealEstateMarketAnalyzer()
    return analyzer.predict(data)


def predict_real_estate_investment(data):
    """Main function for real estate investment scoring"""
    scorer = RealEstateInvestmentScorer()
    return scorer.predict(data)