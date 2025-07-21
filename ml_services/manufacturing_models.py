"""
Manufacturing Industry ML Models
Specialized models for manufacturing including quality prediction, maintenance forecasting, and demand planning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
import logging


class ManufacturingQualityPredictor:
    """Quality prediction for manufacturing processes"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_manufacturing_data(self, data):
        """Manufacturing-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Manufacturing feature engineering
        features = []
        
        # Process parameters
        process_features = ['temperature', 'pressure', 'humidity', 'speed', 'feed_rate',
                          'cutting_speed', 'material_grade', 'tool_wear', 'vibration_level']
        for feature in process_features:
            if feature in df.columns:
                features.append(feature)
                
        # Material properties
        material_features = ['material_hardness', 'material_density', 'material_composition',
                           'batch_number', 'supplier_quality_score', 'material_age_days']
        for feature in material_features:
            if feature in df.columns:
                features.append(feature)
                
        # Equipment status
        equipment_features = ['machine_age', 'maintenance_hours_since', 'calibration_status',
                            'operator_experience', 'shift_type', 'production_volume']
        for feature in equipment_features:
            if feature in df.columns:
                features.append(feature)
        
        # Environmental conditions
        environment_features = ['ambient_temperature', 'ambient_humidity', 'air_quality_index',
                              'noise_level', 'lighting_condition']
        for feature in environment_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with manufacturing logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                # Use process control limits for missing values
                if 'temperature' in col.lower():
                    df[col] = df[col].fillna(25)  # Room temperature default
                elif 'pressure' in col.lower():
                    df[col] = df[col].fillna(1)   # Atmospheric pressure
                elif 'score' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 3)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Predict manufacturing quality"""
        try:
            df, features = self.preprocess_manufacturing_data(data)
            
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
            
            quality_proba = self.model.predict_proba(X_scaled)[0]
            quality_prediction = self.model.predict(X_scaled)[0]
            
            # Quality analysis
            quality_score = quality_proba[1] if len(quality_proba) > 1 else 0.85
            
            return {
                'quality_prediction': int(quality_prediction),
                'quality_score': float(quality_score),
                'quality_grade': self._categorize_quality(quality_score),
                'confidence': float(max(quality_proba)),
                'risk_factors': self._identify_quality_risks(df.iloc[0]),
                'process_recommendations': self._recommend_process_adjustments(df.iloc[0]),
                'control_limits': self._calculate_control_limits(df.iloc[0]),
                'defect_probability': float(1 - quality_score),
                'next_inspection_recommendation': self._recommend_inspection_frequency(quality_score)
            }
            
        except Exception as e:
            logging.error(f"Manufacturing quality prediction error: {str(e)}")
            return {
                'quality_prediction': 1,
                'quality_score': 0.85,
                'quality_grade': 'acceptable',
                'confidence': 0.7,
                'error': f'Prediction error: {str(e)}',
                'risk_factors': ['data_quality_issues']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic manufacturing data"""
        np.random.seed(42)
        n_samples = 1500
        
        X = np.random.randn(n_samples, len(features))
        
        # Quality logic: temperature control, tool wear, material grade
        y = np.zeros(n_samples)
        for i in range(n_samples):
            quality_score = (abs(X[i, 0]) * -0.3 +    # temperature deviation (bad)
                           X[i, 1] * -0.4 +           # tool wear (bad)
                           X[i, 2] * 0.3 +            # material grade (good)
                           np.random.randn() * 0.1)
            y[i] = 1 if quality_score > -0.2 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_quality(self, score):
        """Categorize quality level"""
        if score >= 0.95:
            return 'excellent'
        elif score >= 0.85:
            return 'good'
        elif score >= 0.7:
            return 'acceptable'
        else:
            return 'below_standard'
    
    def _identify_quality_risks(self, process_data):
        """Identify quality risk factors"""
        risks = []
        
        if process_data.get('tool_wear', 0) > 0.8:
            risks.append('high_tool_wear')
        if abs(process_data.get('temperature', 25) - 25) > 5:
            risks.append('temperature_deviation')
        if process_data.get('maintenance_hours_since', 0) > 200:
            risks.append('overdue_maintenance')
        if process_data.get('operator_experience', 5) < 2:
            risks.append('inexperienced_operator')
        if process_data.get('vibration_level', 1) > 3:
            risks.append('excessive_vibration')
        
        return risks or ['nominal_conditions']
    
    def _recommend_process_adjustments(self, process_data):
        """Recommend process parameter adjustments"""
        recommendations = []
        
        temp = process_data.get('temperature', 25)
        if temp > 30:
            recommendations.append('reduce_temperature')
        elif temp < 20:
            recommendations.append('increase_temperature')
        
        if process_data.get('tool_wear', 0) > 0.7:
            recommendations.append('schedule_tool_replacement')
        
        if process_data.get('speed', 100) > 120:
            recommendations.append('reduce_processing_speed')
        
        return recommendations or ['maintain_current_parameters']
    
    def _calculate_control_limits(self, process_data):
        """Calculate statistical process control limits"""
        # Simplified control limits calculation
        return {
            'temperature_ucl': 30,
            'temperature_lcl': 20,
            'pressure_ucl': 5,
            'pressure_lcl': 0.5,
            'vibration_ucl': 2.5
        }
    
    def _recommend_inspection_frequency(self, quality_score):
        """Recommend inspection frequency based on quality"""
        if quality_score >= 0.95:
            return 'standard_sampling'
        elif quality_score >= 0.85:
            return 'increased_sampling'
        else:
            return 'continuous_monitoring'


class PredictiveMaintenanceModel:
    """Predictive maintenance for manufacturing equipment"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def predict(self, data):
        """Predict maintenance requirements"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Maintenance prediction features
            maintenance_features = ['operating_hours', 'temperature_avg', 'vibration_rms', 'pressure_variance',
                                  'power_consumption', 'oil_temperature', 'bearing_temperature', 'rpm_variation',
                                  'last_maintenance_hours', 'fault_history_count', 'load_factor']
            
            available_features = [f for f in maintenance_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy models if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_maintenance_models(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                
                # Predict remaining useful life
                rul_prediction = self.model.predict(X_scaled)[0]
                
                # Detect anomalies
                anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
            else:
                rul_prediction = 720  # 30 days default
                anomaly_score = 0.5
                is_anomaly = False
            
            return {
                'remaining_useful_life_hours': float(max(0, rul_prediction)),
                'maintenance_urgency': self._categorize_maintenance_urgency(rul_prediction),
                'anomaly_detected': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'health_index': self._calculate_health_index(df.iloc[0], rul_prediction),
                'failure_risk_factors': self._identify_failure_risks(df.iloc[0]),
                'maintenance_recommendations': self._recommend_maintenance_actions(rul_prediction, df.iloc[0]),
                'cost_analysis': self._analyze_maintenance_costs(rul_prediction, df.iloc[0]),
                'optimal_maintenance_window': self._suggest_maintenance_window(rul_prediction)
            }
            
        except Exception as e:
            return {
                'remaining_useful_life_hours': 720,
                'maintenance_urgency': 'routine',
                'anomaly_detected': False,
                'health_index': 80,
                'error': str(e)
            }
    
    def _train_maintenance_models(self, features):
        """Train maintenance prediction models"""
        np.random.seed(42)
        n_samples = 1200
        
        X = np.random.randn(n_samples, len(features))
        
        # RUL prediction logic: operating hours, vibration, temperature
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_life = 1000
            wear_factor = X[i, 0] * -50 if len(features) > 0 else 0      # operating hours
            vibration_impact = X[i, 1] * -30 if len(features) > 1 else 0  # vibration
            temp_impact = X[i, 2] * -20 if len(features) > 2 else 0       # temperature
            
            y[i] = max(0, base_life + wear_factor + vibration_impact + temp_impact + np.random.randn() * 50)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.anomaly_detector.fit(X_scaled)
    
    def _categorize_maintenance_urgency(self, rul_hours):
        """Categorize maintenance urgency"""
        if rul_hours < 24:
            return 'critical'
        elif rul_hours < 72:
            return 'urgent'
        elif rul_hours < 168:  # 1 week
            return 'scheduled'
        else:
            return 'routine'
    
    def _calculate_health_index(self, equipment_data, rul_hours):
        """Calculate equipment health index (0-100)"""
        base_health = min(100, (rul_hours / 1000) * 100)
        
        # Adjust for operating conditions
        if equipment_data.get('vibration_rms', 1) > 3:
            base_health -= 10
        if equipment_data.get('temperature_avg', 25) > 80:
            base_health -= 15
        if equipment_data.get('oil_temperature', 60) > 90:
            base_health -= 10
        
        return max(0, int(base_health))
    
    def _identify_failure_risks(self, equipment_data):
        """Identify equipment failure risk factors"""
        risks = []
        
        if equipment_data.get('vibration_rms', 1) > 4:
            risks.append('excessive_vibration')
        if equipment_data.get('temperature_avg', 25) > 85:
            risks.append('overheating')
        if equipment_data.get('oil_temperature', 60) > 95:
            risks.append('lubrication_issues')
        if equipment_data.get('power_consumption', 100) > 120:
            risks.append('increased_power_draw')
        if equipment_data.get('last_maintenance_hours', 0) > 500:
            risks.append('overdue_maintenance')
        
        return risks or ['normal_operation']
    
    def _recommend_maintenance_actions(self, rul_hours, equipment_data):
        """Recommend specific maintenance actions"""
        actions = []
        
        if rul_hours < 72:
            actions.append('emergency_inspection')
        
        if equipment_data.get('vibration_rms', 1) > 3:
            actions.append('bearing_inspection')
        
        if equipment_data.get('oil_temperature', 60) > 85:
            actions.append('lubrication_system_check')
        
        if equipment_data.get('last_maintenance_hours', 0) > 300:
            actions.append('routine_maintenance_overdue')
        
        return actions or ['continue_monitoring']
    
    def _analyze_maintenance_costs(self, rul_hours, equipment_data):
        """Analyze maintenance cost implications"""
        if rul_hours < 24:
            return {
                'emergency_cost_multiplier': 3.0,
                'downtime_cost_estimate': 5000,
                'replacement_probability': 0.3
            }
        elif rul_hours < 168:
            return {
                'planned_maintenance_cost': 1500,
                'downtime_cost_estimate': 1000,
                'replacement_probability': 0.1
            }
        else:
            return {
                'routine_maintenance_cost': 500,
                'downtime_cost_estimate': 200,
                'replacement_probability': 0.02
            }
    
    def _suggest_maintenance_window(self, rul_hours):
        """Suggest optimal maintenance timing"""
        if rul_hours < 24:
            return 'immediate'
        elif rul_hours < 72:
            return 'within_24_hours'
        elif rul_hours < 168:
            return 'within_week'
        else:
            return 'next_scheduled_maintenance'


class ManufacturingDemandPlanner:
    """Demand planning for manufacturing production"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Predict manufacturing demand"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Demand planning features
            demand_features = ['historical_demand_12m', 'seasonal_index', 'economic_indicator',
                             'customer_orders_pipeline', 'inventory_level', 'lead_time_days',
                             'market_growth_rate', 'competitor_capacity', 'raw_material_availability']
            
            available_features = [f for f in demand_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_demand_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                demand_forecast = self.model.predict(X_scaled)[0]
            else:
                demand_forecast = df.iloc[0].get('historical_demand_12m', 1000) * 1.05
            
            return {
                'demand_forecast': float(max(0, demand_forecast)),
                'forecast_period': 'next_quarter',
                'capacity_utilization': self._calculate_capacity_utilization(demand_forecast, df.iloc[0]),
                'production_recommendations': self._recommend_production_strategy(demand_forecast, df.iloc[0]),
                'supply_chain_impact': self._assess_supply_chain_impact(demand_forecast, df.iloc[0]),
                'risk_factors': self._identify_demand_risks(df.iloc[0]),
                'inventory_strategy': self._recommend_inventory_strategy(demand_forecast, df.iloc[0])
            }
            
        except Exception as e:
            return {
                'demand_forecast': 1000,
                'forecast_period': 'next_quarter',
                'capacity_utilization': 75,
                'error': str(e)
            }
    
    def _train_demand_model(self, features):
        """Train demand forecasting model"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, len(features))
        
        # Demand logic: historical trends, seasonality, economic factors
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_demand = 1000
            trend = X[i, 0] * 100 if len(features) > 0 else 0
            seasonal = X[i, 1] * 200 if len(features) > 1 else 0
            economic = X[i, 2] * 150 if len(features) > 2 else 0
            
            y[i] = max(0, base_demand + trend + seasonal + economic + np.random.randn() * 50)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _calculate_capacity_utilization(self, demand_forecast, data):
        """Calculate expected capacity utilization"""
        max_capacity = data.get('max_production_capacity', demand_forecast * 1.2)
        utilization = (demand_forecast / max_capacity) * 100
        return min(100, max(0, utilization))
    
    def _recommend_production_strategy(self, demand_forecast, data):
        """Recommend production strategy"""
        current_capacity = data.get('current_production_capacity', demand_forecast * 0.8)
        
        if demand_forecast > current_capacity * 1.1:
            return 'increase_production_capacity'
        elif demand_forecast < current_capacity * 0.7:
            return 'optimize_production_efficiency'
        else:
            return 'maintain_current_production'
    
    def _assess_supply_chain_impact(self, demand_forecast, data):
        """Assess supply chain implications"""
        lead_time = data.get('lead_time_days', 30)
        raw_material_availability = data.get('raw_material_availability', 0.9)
        
        if demand_forecast > data.get('historical_demand_12m', 1000) * 1.2:
            if raw_material_availability < 0.8:
                return 'supply_constraints_expected'
            elif lead_time > 45:
                return 'extended_lead_times_likely'
            else:
                return 'manageable_with_planning'
        else:
            return 'normal_supply_chain_operations'
    
    def _identify_demand_risks(self, data):
        """Identify demand planning risks"""
        risks = []
        
        if data.get('market_growth_rate', 0.05) < 0:
            risks.append('market_decline')
        if data.get('competitor_capacity', 1000) > data.get('historical_demand_12m', 1000) * 2:
            risks.append('oversupply_risk')
        if data.get('raw_material_availability', 0.9) < 0.7:
            risks.append('material_shortage')
        if data.get('economic_indicator', 1.0) < 0.95:
            risks.append('economic_downturn')
        
        return risks or ['standard_market_conditions']
    
    def _recommend_inventory_strategy(self, demand_forecast, data):
        """Recommend inventory management strategy"""
        current_inventory = data.get('inventory_level', demand_forecast * 0.2)
        
        if demand_forecast > current_inventory * 6:  # 2 months supply
            return 'increase_safety_stock'
        elif demand_forecast < current_inventory * 2:  # 6 months supply
            return 'reduce_inventory_levels'
        else:
            return 'maintain_inventory_strategy'


def predict_manufacturing_quality(data):
    """Main function for manufacturing quality prediction"""
    predictor = ManufacturingQualityPredictor()
    return predictor.predict(data)


def predict_maintenance_needs(data):
    """Main function for predictive maintenance"""
    maintenance_model = PredictiveMaintenanceModel()
    return maintenance_model.predict(data)


def predict_manufacturing_demand(data):
    """Main function for manufacturing demand planning"""
    planner = ManufacturingDemandPlanner()
    return planner.predict(data)