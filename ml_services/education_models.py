"""
Education Industry ML Models
Specialized models for educational institutions including student retention, performance prediction, and engagement analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime, timedelta
import logging


class StudentRetentionPredictor:
    """Predict student retention and dropout risk"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_education_data(self, data):
        """Education-specific data preprocessing"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Education feature engineering
        features = []
        
        # Academic performance
        academic_features = ['gpa', 'attendance_rate', 'assignment_completion_rate', 'test_scores_avg',
                           'credit_hours_attempted', 'credit_hours_completed', 'course_difficulty_avg',
                           'failed_courses_count', 'repeated_courses_count', 'academic_standing']
        for feature in academic_features:
            if feature in df.columns:
                features.append(feature)
                
        # Engagement metrics
        engagement_features = ['lms_login_frequency', 'discussion_posts_count', 'office_hours_visits',
                             'library_usage_hours', 'campus_activity_participation', 'study_group_participation',
                             'tutoring_sessions_attended', 'online_resource_usage']
        for feature in engagement_features:
            if feature in df.columns:
                features.append(feature)
                
        # Financial factors
        financial_features = ['financial_aid_amount', 'tuition_balance', 'work_study_hours',
                            'external_employment_hours', 'family_income_bracket', 'scholarship_amount']
        for feature in financial_features:
            if feature in df.columns:
                features.append(feature)
        
        # Personal and demographic
        personal_features = ['age', 'distance_from_home', 'first_generation_college', 'housing_type',
                           'commute_time_minutes', 'family_support_score', 'mental_health_services_usage']
        for feature in personal_features:
            if feature in df.columns:
                features.append(feature)
        
        # Handle missing values with educational logic
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('unknown')
            else:
                if 'rate' in col.lower() or 'percentage' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0.75)
                elif 'gpa' in col.lower():
                    df[col] = df[col].fillna(2.5)  # Below average GPA as risk indicator
                elif 'score' in col.lower():
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 3)
                else:
                    df[col] = df[col].fillna(0)
        
        return df, features
    
    def predict(self, data):
        """Predict student retention probability"""
        try:
            df, features = self.preprocess_education_data(data)
            
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
            
            retention_proba = self.model.predict_proba(X_scaled)[0]
            retention_prediction = self.model.predict(X_scaled)[0]
            
            retention_probability = retention_proba[1] if len(retention_proba) > 1 else retention_proba[0]
            
            return {
                'retention_prediction': int(retention_prediction),
                'retention_probability': float(retention_probability),
                'dropout_risk': self._categorize_dropout_risk(1 - retention_probability),
                'confidence': float(max(retention_proba)),
                'academic_health_score': self._calculate_academic_health(df.iloc[0]),
                'risk_factors': self._identify_retention_risks(df.iloc[0]),
                'intervention_recommendations': self._recommend_interventions(1 - retention_probability, df.iloc[0]),
                'support_services_needed': self._identify_support_needs(df.iloc[0]),
                'early_warning_indicators': self._detect_early_warnings(df.iloc[0])
            }
            
        except Exception as e:
            logging.error(f"Student retention prediction error: {str(e)}")
            return {
                'retention_prediction': 1,
                'retention_probability': 0.8,
                'dropout_risk': 'low',
                'confidence': 0.7,
                'error': f'Prediction error: {str(e)}',
                'risk_factors': ['data_processing_error']
            }
    
    def _train_dummy_model(self, features):
        """Train with synthetic education data"""
        np.random.seed(42)
        n_samples = 1800
        
        X = np.random.randn(n_samples, len(features))
        
        # Retention logic: GPA, attendance, engagement, financial stress
        y = np.zeros(n_samples)
        for i in range(n_samples):
            retention_score = (X[i, 0] * 0.4 +      # GPA/academic performance
                             X[i, 1] * 0.3 +        # attendance/engagement
                             X[i, 2] * -0.2 +       # financial stress (negative)
                             np.random.randn() * 0.1)
            y[i] = 1 if retention_score > -0.1 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_dropout_risk(self, dropout_probability):
        """Categorize dropout risk level"""
        if dropout_probability < 0.15:
            return 'low'
        elif dropout_probability < 0.35:
            return 'moderate'
        elif dropout_probability < 0.65:
            return 'high'
        else:
            return 'critical'
    
    def _calculate_academic_health(self, student_data):
        """Calculate academic health score (0-100)"""
        score = 50  # Base score
        
        # Academic performance factors
        gpa = student_data.get('gpa', 2.5)
        score += (gpa - 2.0) * 15  # Scale GPA contribution
        
        attendance = student_data.get('attendance_rate', 0.8)
        score += (attendance - 0.5) * 40  # Attendance impact
        
        completion = student_data.get('assignment_completion_rate', 0.8)
        score += (completion - 0.5) * 30  # Assignment completion
        
        # Negative factors
        if student_data.get('failed_courses_count', 0) > 1:
            score -= 15
        if student_data.get('academic_standing', 'good') == 'probation':
            score -= 25
        
        return max(0, min(100, int(score)))
    
    def _identify_retention_risks(self, student_data):
        """Identify student retention risk factors"""
        risks = []
        
        if student_data.get('gpa', 3.0) < 2.5:
            risks.append('low_academic_performance')
        if student_data.get('attendance_rate', 0.9) < 0.75:
            risks.append('poor_attendance')
        if student_data.get('assignment_completion_rate', 0.9) < 0.7:
            risks.append('incomplete_assignments')
        if student_data.get('failed_courses_count', 0) > 1:
            risks.append('course_failures')
        if student_data.get('tuition_balance', 0) > 5000:
            risks.append('financial_burden')
        if student_data.get('lms_login_frequency', 5) < 2:
            risks.append('low_engagement')
        if student_data.get('office_hours_visits', 2) == 0:
            risks.append('lack_of_help_seeking')
        if student_data.get('distance_from_home', 50) > 500:
            risks.append('homesickness_risk')
        
        return risks or ['standard_risk_profile']
    
    def _recommend_interventions(self, dropout_risk, student_data):
        """Recommend intervention strategies"""
        if dropout_risk < 0.15:
            return ['academic_enrichment_opportunities', 'leadership_development']
        elif dropout_risk < 0.35:
            return ['academic_coaching', 'study_skills_workshop', 'peer_mentoring']
        elif dropout_risk < 0.65:
            return ['intensive_academic_support', 'financial_aid_counseling', 'counseling_services']
        else:
            return ['emergency_intervention', 'academic_probation_support', 'retention_specialist_assignment']
    
    def _identify_support_needs(self, student_data):
        """Identify specific support service needs"""
        needs = []
        
        if student_data.get('gpa', 3.0) < 2.5:
            needs.append('academic_tutoring')
        if student_data.get('tuition_balance', 0) > 3000:
            needs.append('financial_counseling')
        if student_data.get('mental_health_services_usage', 0) > 0:
            needs.append('counseling_support')
        if student_data.get('work_study_hours', 0) + student_data.get('external_employment_hours', 0) > 25:
            needs.append('time_management_coaching')
        if student_data.get('first_generation_college', False):
            needs.append('first_gen_support_program')
        
        return needs or ['general_student_support']
    
    def _detect_early_warnings(self, student_data):
        """Detect early warning indicators"""
        warnings = []
        
        if student_data.get('attendance_rate', 0.9) < 0.8:
            warnings.append('attendance_declining')
        if student_data.get('lms_login_frequency', 5) < 3:
            warnings.append('engagement_dropping')
        if student_data.get('assignment_completion_rate', 0.9) < 0.8:
            warnings.append('assignments_slipping')
        if student_data.get('test_scores_avg', 80) < 70:
            warnings.append('test_performance_declining')
        
        return warnings or ['no_immediate_warnings']


class StudentPerformancePredictor:
    """Predict student academic performance"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Predict student performance metrics"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Performance prediction features
            performance_features = ['current_gpa', 'study_hours_per_week', 'class_attendance_rate',
                                  'previous_semester_gpa', 'course_load_credit_hours', 'prerequisite_gpa',
                                  'participation_score', 'homework_submission_rate', 'quiz_scores_avg']
            
            available_features = [f for f in performance_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_performance_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                predicted_gpa = self.model.predict(X_scaled)[0]
            else:
                predicted_gpa = df.iloc[0].get('current_gpa', 3.0)
            
            # Ensure GPA is within valid range
            predicted_gpa = max(0.0, min(4.0, predicted_gpa))
            
            return {
                'predicted_gpa': float(predicted_gpa),
                'performance_trend': self._determine_performance_trend(df.iloc[0], predicted_gpa),
                'grade_prediction': self._predict_letter_grade(predicted_gpa),
                'improvement_potential': self._assess_improvement_potential(df.iloc[0]),
                'study_recommendations': self._recommend_study_strategies(df.iloc[0], predicted_gpa),
                'course_success_probability': self._calculate_success_probability(predicted_gpa),
                'academic_goals_alignment': self._assess_goals_alignment(predicted_gpa)
            }
            
        except Exception as e:
            return {
                'predicted_gpa': 3.0,
                'performance_trend': 'stable',
                'grade_prediction': 'B',
                'error': str(e)
            }
    
    def _train_performance_model(self, features):
        """Train performance prediction model"""
        np.random.seed(42)
        n_samples = 1500
        
        X = np.random.randn(n_samples, len(features))
        
        # Performance logic: study hours, attendance, previous performance
        y = np.zeros(n_samples)
        for i in range(n_samples):
            base_gpa = 2.5
            study_effect = X[i, 0] * 0.3 if len(features) > 0 else 0
            attendance_effect = X[i, 1] * 0.4 if len(features) > 1 else 0
            previous_effect = X[i, 2] * 0.5 if len(features) > 2 else 0
            
            predicted_gpa = base_gpa + study_effect + attendance_effect + previous_effect + np.random.randn() * 0.2
            y[i] = max(0.0, min(4.0, predicted_gpa))
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _determine_performance_trend(self, student_data, predicted_gpa):
        """Determine academic performance trend"""
        current_gpa = student_data.get('current_gpa', 3.0)
        
        if predicted_gpa > current_gpa + 0.2:
            return 'improving'
        elif predicted_gpa < current_gpa - 0.2:
            return 'declining'
        else:
            return 'stable'
    
    def _predict_letter_grade(self, gpa):
        """Convert GPA to letter grade prediction"""
        if gpa >= 3.7:
            return 'A'
        elif gpa >= 3.3:
            return 'B+'
        elif gpa >= 3.0:
            return 'B'
        elif gpa >= 2.7:
            return 'B-'
        elif gpa >= 2.3:
            return 'C+'
        elif gpa >= 2.0:
            return 'C'
        else:
            return 'Below C'
    
    def _assess_improvement_potential(self, student_data):
        """Assess potential for academic improvement"""
        study_hours = student_data.get('study_hours_per_week', 10)
        attendance = student_data.get('class_attendance_rate', 0.85)
        
        if study_hours < 15 and attendance < 0.9:
            return 'high_potential'
        elif study_hours < 20 or attendance < 0.95:
            return 'moderate_potential'
        else:
            return 'limited_potential'
    
    def _recommend_study_strategies(self, student_data, predicted_gpa):
        """Recommend study strategies"""
        strategies = []
        
        if student_data.get('study_hours_per_week', 10) < 15:
            strategies.append('increase_study_time')
        if student_data.get('class_attendance_rate', 0.9) < 0.9:
            strategies.append('improve_class_attendance')
        if student_data.get('homework_submission_rate', 0.9) < 0.95:
            strategies.append('better_assignment_management')
        if predicted_gpa < 3.0:
            strategies.append('seek_tutoring_support')
        
        return strategies or ['maintain_current_habits']
    
    def _calculate_success_probability(self, predicted_gpa):
        """Calculate probability of course success"""
        if predicted_gpa >= 3.5:
            return 0.95
        elif predicted_gpa >= 3.0:
            return 0.85
        elif predicted_gpa >= 2.5:
            return 0.70
        elif predicted_gpa >= 2.0:
            return 0.55
        else:
            return 0.35
    
    def _assess_goals_alignment(self, predicted_gpa):
        """Assess alignment with academic goals"""
        if predicted_gpa >= 3.7:
            return 'exceeds_expectations'
        elif predicted_gpa >= 3.0:
            return 'meets_expectations'
        elif predicted_gpa >= 2.5:
            return 'below_expectations'
        else:
            return 'significant_intervention_needed'


class StudentEngagementAnalyzer:
    """Analyze and predict student engagement levels"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def predict(self, data):
        """Analyze student engagement patterns"""
        try:
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            
            # Engagement features
            engagement_features = ['lms_daily_logins', 'forum_participation_score', 'video_completion_rate',
                                 'assignment_early_submission_rate', 'office_hours_attendance', 'peer_interaction_score',
                                 'resource_download_frequency', 'quiz_attempt_timing', 'collaboration_tool_usage']
            
            available_features = [f for f in engagement_features if f in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Train dummy model if needed
            if not hasattr(self.model, 'feature_importances_'):
                self._train_engagement_model(available_features)
            
            X = df[available_features].fillna(0)
            
            # Handle categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if len(X.columns) > 0:
                X_scaled = self.scaler.transform(X)
                engagement_proba = self.model.predict_proba(X_scaled)[0]
                engagement_prediction = self.model.predict(X_scaled)[0]
            else:
                engagement_proba = [0.4, 0.6]
                engagement_prediction = 1
            
            engagement_score = engagement_proba[1] if len(engagement_proba) > 1 else 0.6
            
            return {
                'engagement_score': float(engagement_score),
                'engagement_level': self._categorize_engagement(engagement_score),
                'engagement_trends': self._analyze_engagement_trends(df.iloc[0]),
                'motivation_indicators': self._assess_motivation(df.iloc[0]),
                'engagement_recommendations': self._recommend_engagement_strategies(engagement_score, df.iloc[0]),
                'social_learning_score': self._calculate_social_learning(df.iloc[0]),
                'digital_literacy_assessment': self._assess_digital_skills(df.iloc[0])
            }
            
        except Exception as e:
            return {
                'engagement_score': 0.6,
                'engagement_level': 'moderate',
                'engagement_trends': 'stable',
                'error': str(e)
            }
    
    def _train_engagement_model(self, features):
        """Train engagement prediction model"""
        np.random.seed(42)
        n_samples = 1200
        
        X = np.random.randn(n_samples, len(features))
        
        # Engagement logic: active participation, consistent usage, collaboration
        y = np.zeros(n_samples)
        for i in range(n_samples):
            engagement_score = (X[i, 0] * 0.3 +     # login frequency
                              X[i, 1] * 0.4 +       # participation
                              X[i, 2] * 0.2)        # collaboration
            y[i] = 1 if engagement_score > 0.1 else 0
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def _categorize_engagement(self, score):
        """Categorize engagement level"""
        if score >= 0.8:
            return 'highly_engaged'
        elif score >= 0.6:
            return 'moderately_engaged'
        elif score >= 0.4:
            return 'partially_engaged'
        else:
            return 'disengaged'
    
    def _analyze_engagement_trends(self, student_data):
        """Analyze engagement trend patterns"""
        login_freq = student_data.get('lms_daily_logins', 3)
        participation = student_data.get('forum_participation_score', 5)
        
        if login_freq > 5 and participation > 7:
            return 'increasing_engagement'
        elif login_freq < 2 or participation < 3:
            return 'declining_engagement'
        else:
            return 'stable_engagement'
    
    def _assess_motivation(self, student_data):
        """Assess student motivation indicators"""
        indicators = []
        
        if student_data.get('assignment_early_submission_rate', 0.5) > 0.8:
            indicators.append('high_intrinsic_motivation')
        if student_data.get('office_hours_attendance', 2) > 3:
            indicators.append('help_seeking_behavior')
        if student_data.get('video_completion_rate', 0.7) > 0.9:
            indicators.append('content_engagement')
        if student_data.get('peer_interaction_score', 5) > 7:
            indicators.append('collaborative_learning')
        
        return indicators or ['standard_motivation_level']
    
    def _recommend_engagement_strategies(self, engagement_score, student_data):
        """Recommend strategies to improve engagement"""
        if engagement_score >= 0.8:
            return ['peer_mentoring_opportunities', 'advanced_projects']
        elif engagement_score >= 0.6:
            return ['gamification_elements', 'group_activities']
        elif engagement_score >= 0.4:
            return ['personalized_content', 'interactive_assignments', 'regular_check_ins']
        else:
            return ['immediate_intervention', 'one_on_one_support', 'alternative_learning_formats']
    
    def _calculate_social_learning(self, student_data):
        """Calculate social learning engagement score"""
        score = 0
        
        score += min(25, student_data.get('forum_participation_score', 0) * 5)
        score += min(25, student_data.get('peer_interaction_score', 0) * 5)
        score += min(25, student_data.get('collaboration_tool_usage', 0) * 5)
        score += min(25, student_data.get('office_hours_attendance', 0) * 8)
        
        return min(100, score)
    
    def _assess_digital_skills(self, student_data):
        """Assess digital literacy and platform usage skills"""
        lms_usage = student_data.get('lms_daily_logins', 3)
        resource_usage = student_data.get('resource_download_frequency', 2)
        collab_usage = student_data.get('collaboration_tool_usage', 3)
        
        if lms_usage > 5 and resource_usage > 4 and collab_usage > 4:
            return 'advanced_digital_skills'
        elif lms_usage > 3 and resource_usage > 2:
            return 'proficient_digital_skills'
        else:
            return 'developing_digital_skills'


def predict_student_retention(data):
    """Main function for student retention prediction"""
    predictor = StudentRetentionPredictor()
    return predictor.predict(data)


def predict_student_performance(data):
    """Main function for student performance prediction"""
    performance_predictor = StudentPerformancePredictor()
    return performance_predictor.predict(data)


def predict_student_engagement(data):
    """Main function for student engagement analysis"""
    engagement_analyzer = StudentEngagementAnalyzer()
    return engagement_analyzer.predict(data)