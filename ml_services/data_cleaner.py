"""
Advanced Data Cleaning and Preprocessing Utilities

This module handles messy, uncleaned data and provides robust preprocessing
for all ML models in the platform.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings


class DataCleaner:
    """
    Comprehensive data cleaning utility that handles:
    - Missing values
    - Inconsistent formats
    - Outliers
    - Text normalization
    - Data type conversion
    """
    
    def __init__(self):
        self.cleaning_report = {}
        self.warnings = []
        self.data_quality_score = 0.0
        
    def clean_data(self, data: pd.DataFrame, model_type: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Main cleaning function that processes raw data for ML models
        
        Args:
            data: Raw DataFrame with potentially messy data
            model_type: Type of ML model (lead_score, churn, etc.)
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        self.cleaning_report = {
            'original_rows': len(data),
            'original_columns': len(data.columns),
            'missing_values_filled': 0,
            'outliers_handled': 0,
            'formats_standardized': 0,
            'duplicates_removed': 0,
            'warnings': []
        }
        
        # 1. Handle missing values
        data = self._handle_missing_values(data)
        
        # 2. Standardize data types and formats
        data = self._standardize_formats(data)
        
        # 3. Handle outliers
        data = self._handle_outliers(data)
        
        # 4. Remove/flag duplicates
        data = self._handle_duplicates(data)
        
        # 5. Model-specific cleaning
        if model_type:
            data = self._model_specific_cleaning(data, model_type)
            
        # 6. Calculate data quality score
        self.data_quality_score = self._calculate_quality_score(data)
        self.cleaning_report['data_quality_score'] = self.data_quality_score
        self.cleaning_report['final_rows'] = len(data)
        self.cleaning_report['final_columns'] = len(data.columns)
        
        return data, self.cleaning_report
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle various forms of missing values"""
        original_nulls = data.isnull().sum().sum()
        
        # Define what we consider as missing values
        missing_indicators = ['', ' ', 'null', 'NULL', 'None', 'NONE', 'n/a', 'N/A', 
                             'nan', 'NaN', 'NAN', '#N/A', '#NULL!', 'nil', 'NIL',
                             'unknown', 'Unknown', 'UNKNOWN', 'tbd', 'TBD', 'TBA',
                             'coming soon', 'Coming Soon', 'COMING SOON', '?', '??',
                             'missing', 'Missing', 'MISSING', 'unavailable', 'Unavailable']
        
        # Replace missing indicators with NaN
        data = data.replace(missing_indicators, np.nan)
        
        # Handle different column types
        for column in data.columns:
            if data[column].dtype == 'object':
                # For text columns, fill with 'Other' or most frequent
                if data[column].isnull().sum() > 0:
                    mode_value = data[column].mode()
                    if len(mode_value) > 0 and pd.notna(mode_value.iloc[0]):
                        data[column].fillna(mode_value.iloc[0], inplace=True)
                    else:
                        data[column].fillna('Other', inplace=True)
            else:
                # For numeric columns, fill with median
                if data[column].isnull().sum() > 0:
                    median_value = data[column].median()
                    if pd.notna(median_value):
                        data[column].fillna(median_value, inplace=True)
                    else:
                        data[column].fillna(0, inplace=True)
        
        filled_values = data.isnull().sum().sum()
        self.cleaning_report['missing_values_filled'] = original_nulls - filled_values
        
        if original_nulls > 0:
            self.warnings.append(f"Filled {original_nulls - filled_values} missing values")
            
        return data
    
    def _standardize_formats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize various data formats"""
        formats_changed = 0
        
        for column in data.columns:
            original_values = data[column].copy()
            
            # Handle currency and numeric formats
            if data[column].dtype == 'object':
                # Try to convert currency and numeric strings
                numeric_converted = self._clean_numeric_column(data[column])
                if numeric_converted is not None and not numeric_converted.equals(original_values):
                    data[column] = numeric_converted
                    formats_changed += 1
                    
                # Handle date formats
                date_converted = self._clean_date_column(data[column])
                if date_converted is not None and not date_converted.equals(original_values):
                    data[column] = date_converted
                    formats_changed += 1
                    
                # Handle boolean-like text
                bool_converted = self._clean_boolean_column(data[column])
                if bool_converted is not None and not bool_converted.equals(original_values):
                    data[column] = bool_converted
                    formats_changed += 1
                    
                # Clean text columns
                if data[column].dtype == 'object':
                    data[column] = self._clean_text_column(data[column])
        
        self.cleaning_report['formats_standardized'] = formats_changed
        if formats_changed > 0:
            self.warnings.append(f"Standardized {formats_changed} column formats")
            
        return data
    
    def _clean_numeric_column(self, series: pd.Series) -> Optional[pd.Series]:
        """Clean numeric columns with currency symbols, commas, etc."""
        if series.dtype != 'object':
            return None
            
        # Check if this looks like a numeric column
        sample_values = series.dropna().head(10).astype(str)
        numeric_pattern = r'^[\$\€\£\¥]?[\-\+]?\s*[\d,]+\.?\d*[%]?[kmKM]?$'
        
        numeric_matches = sample_values.str.match(numeric_pattern).sum()
        if numeric_matches < len(sample_values) * 0.5:
            return None  # Not primarily numeric
            
        cleaned = series.astype(str).copy()
        
        # Remove currency symbols
        cleaned = cleaned.str.replace(r'[\$\€\£\¥]', '', regex=True)
        
        # Handle percentage
        is_percentage = cleaned.str.contains('%', na=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        
        # Handle k, m suffixes (1k = 1000, 1m = 1000000)
        k_values = cleaned.str.contains('k|K', na=False)
        m_values = cleaned.str.contains('m|M', na=False)
        cleaned = cleaned.str.replace(r'[kmKM]', '', regex=True)
        
        # Remove commas and whitespace
        cleaned = cleaned.str.replace(',', '').str.strip()
        
        # Convert to numeric
        try:
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            
            # Apply multipliers
            numeric_series.loc[k_values] *= 1000
            numeric_series.loc[m_values] *= 1000000
            numeric_series.loc[is_percentage] /= 100
            
            # If we successfully converted most values, return the cleaned series
            if numeric_series.notna().sum() >= len(series) * 0.5:
                return numeric_series
                
        except Exception:
            pass
            
        return None
    
    def _clean_date_column(self, series: pd.Series) -> Optional[pd.Series]:
        """Clean date columns with various formats"""
        if series.dtype != 'object':
            return None
            
        # Check if this looks like dates
        sample_values = series.dropna().head(5).astype(str)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
        ]
        
        is_date_like = False
        for pattern in date_patterns:
            if sample_values.str.contains(pattern, na=False).any():
                is_date_like = True
                break
                
        if not is_date_like:
            return None
            
        # Try to convert to datetime
        try:
            # Try multiple date formats
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', 
                           '%B %d, %Y', '%b %d, %Y', '%d %B %Y']
            
            for fmt in date_formats:
                try:
                    converted = pd.to_datetime(series, format=fmt, errors='coerce')
                    if converted.notna().sum() >= len(series) * 0.7:
                        return converted
                except:
                    continue
                    
            # If specific formats don't work, try automatic parsing
            converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            if converted.notna().sum() >= len(series) * 0.5:
                return converted
                
        except Exception:
            pass
            
        return None
    
    def _clean_boolean_column(self, series: pd.Series) -> Optional[pd.Series]:
        """Clean boolean-like text columns"""
        if series.dtype != 'object':
            return None
            
        unique_values = set(series.dropna().astype(str).str.lower().unique())
        boolean_indicators = {
            'true': True, 'false': False, 'yes': True, 'no': False,
            'y': True, 'n': False, '1': True, '0': False,
            'on': True, 'off': False, 'active': True, 'inactive': False,
            'enabled': True, 'disabled': False, 'pass': True, 'fail': False
        }
        
        # Check if most values are boolean-like
        boolean_matches = len(unique_values.intersection(boolean_indicators.keys()))
        if boolean_matches >= len(unique_values) * 0.7:
            cleaned = series.astype(str).str.lower().map(boolean_indicators)
            return cleaned
            
        return None
    
    def _clean_text_column(self, series: pd.Series) -> pd.Series:
        """Clean text columns - normalize case, trim, handle special chars"""
        if series.dtype != 'object':
            return series
            
        cleaned = series.astype(str).copy()
        
        # Remove HTML tags
        cleaned = cleaned.str.replace(r'<[^>]+>', '', regex=True)
        
        # Normalize whitespace
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Fix common encoding issues
        cleaned = cleaned.str.replace(r'[''""„"«»]', '"', regex=True)
        cleaned = cleaned.str.replace(r'[–—]', '-', regex=True)
        
        return cleaned
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        outliers_handled = 0
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(data[column].unique()) > 10:  # Skip categorical-like numeric columns
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing them with explicit type conversion
                    lower_mask = data[column] < lower_bound
                    upper_mask = data[column] > upper_bound
                    
                    if lower_mask.any():
                        data.loc[lower_mask, column] = data[column].dtype.type(lower_bound)
                    if upper_mask.any():
                        data.loc[upper_mask, column] = data[column].dtype.type(upper_bound)
                    outliers_handled += outlier_count
                    
        self.cleaning_report['outliers_handled'] = outliers_handled
        if outliers_handled > 0:
            self.warnings.append(f"Capped {outliers_handled} outlier values")
            
        return data
    
    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate records"""
        original_count = len(data)
        
        # Check for exact duplicates
        duplicates = data.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            # Keep first occurrence of duplicates
            data = data.drop_duplicates()
            self.cleaning_report['duplicates_removed'] = duplicate_count
            self.warnings.append(f"Removed {duplicate_count} duplicate rows")
            
        return data
    
    def _model_specific_cleaning(self, data: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Apply model-specific cleaning rules"""
        
        if model_type == 'lead_score':
            data = self._clean_lead_score_data(data)
        elif model_type == 'churn':
            data = self._clean_churn_data(data)
        elif model_type == 'sales_forecast':
            data = self._clean_sales_forecast_data(data)
        elif model_type in ['sentiment', 'keywords']:
            data = self._clean_text_data(data)
            
        return data
    
    def _clean_lead_score_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean lead scoring specific data"""
        # Map company size text to numbers
        if 'company_size' in data.columns:
            size_mapping = {
                'small': 50, 'medium': 250, 'large': 1000,
                'startup': 25, 'enterprise': 5000,
                '1-10': 5, '10-50': 30, '50-200': 125, '200-1000': 600,
                '1000+': 2000, 'micro': 10, 'smb': 100
            }
            
            # Extract numbers from text like "50-100 employees"
            data['company_size'] = data['company_size'].astype(str)
            for key, value in size_mapping.items():
                mask = data['company_size'].str.contains(key, case=False, na=False)
                data.loc[mask, 'company_size'] = value
                
        # Normalize scores to 1-5 range if needed
        score_columns = ['industry_score', 'engagement_score', 'demographic_score', 'behavioral_score']
        for col in score_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # If scores are outside 1-5 range, normalize them
                if data[col].max() > 5 or data[col].min() < 1:
                    data[col] = ((data[col] - data[col].min()) / (data[col].max() - data[col].min()) * 4) + 1
                    
        return data
    
    def _clean_churn_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean churn prediction specific data"""
        # Convert tenure text to months
        if 'customer_tenure' in data.columns:
            data['customer_tenure'] = data['customer_tenure'].astype(str)
            
            # Convert "2 years" to 24, "6 months" to 6, etc.
            year_pattern = r'(\d+)\s*year'
            month_pattern = r'(\d+)\s*month'
            
            years = data['customer_tenure'].str.extract(year_pattern, expand=False).astype(float) * 12
            months = data['customer_tenure'].str.extract(month_pattern, expand=False).astype(float)
            
            # Combine and fill missing
            data['customer_tenure'] = years.fillna(0) + months.fillna(0)
            data['customer_tenure'] = data['customer_tenure'].replace(0, np.nan)
            data['customer_tenure'] = pd.to_numeric(data['customer_tenure'], errors='coerce')
            
        return data
    
    def _clean_sales_forecast_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean sales forecast specific data"""
        # Ensure date column is properly formatted
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            
        # Handle sales amount with currency
        if 'sales_amount' in data.columns:
            data['sales_amount'] = self._clean_numeric_column(data['sales_amount'])
            if data['sales_amount'] is None:
                data['sales_amount'] = pd.to_numeric(data['sales_amount'], errors='coerce')
                
        return data
    
    def _clean_text_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean text data for NLP models"""
        text_columns = ['text', 'review', 'comment', 'description', 'content']
        
        for col in text_columns:
            if col in data.columns:
                # Remove very short texts (less than 3 characters)
                data = data[data[col].astype(str).str.len() >= 3]
                
                # Clean text content
                data[col] = data[col].astype(str)
                data[col] = data[col].str.replace(r'http\S+', '', regex=True)  # Remove URLs
                data[col] = data[col].str.replace(r'@\w+', '', regex=True)     # Remove mentions
                data[col] = data[col].str.replace(r'#\w+', '', regex=True)     # Remove hashtags
                
        return data
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-1)"""
        scores = []
        
        # Completeness score (no missing values)
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        scores.append(completeness * 0.4)  # 40% weight
        
        # Consistency score (similar data types in columns)
        consistency = 0.8  # Assume reasonable after cleaning
        scores.append(consistency * 0.3)  # 30% weight
        
        # Validity score (values in expected ranges)
        validity = 0.7  # Conservative estimate after outlier handling
        scores.append(validity * 0.2)  # 20% weight
        
        # Uniqueness score (not too many duplicates)
        total_rows = len(data)
        unique_rows = len(data.drop_duplicates())
        uniqueness = unique_rows / total_rows if total_rows > 0 else 0
        scores.append(uniqueness * 0.1)  # 10% weight
        
        return sum(scores)
    
    def get_cleaning_summary(self) -> Dict:
        """Get a summary of all cleaning operations performed"""
        summary = self.cleaning_report.copy()
        summary['warnings'] = self.warnings
        summary['recommendation'] = self._get_recommendation()
        return summary
    
    def _get_recommendation(self) -> str:
        """Provide recommendation based on data quality"""
        if self.data_quality_score >= 0.9:
            return "Excellent data quality! Predictions should be highly accurate."
        elif self.data_quality_score >= 0.7:
            return "Good data quality. Minor improvements possible but predictions should be reliable."
        elif self.data_quality_score >= 0.5:
            return "Fair data quality. Consider cleaning more data for better prediction accuracy."
        else:
            return "Poor data quality detected. Strong recommendation to clean data before processing for optimal results."


def clean_api_data(data: Dict[str, Any], model_type: str = None) -> Dict[str, Any]:
    """
    Clean individual API request data (single prediction)
    
    Args:
        data: Dictionary of input data from API request
        model_type: Type of ML model
        
    Returns:
        Cleaned data dictionary
    """
    cleaner = DataCleaner()
    
    # Convert to DataFrame for consistent processing
    df = pd.DataFrame([data])
    cleaned_df, report = cleaner.clean_data(df, model_type)
    
    # Convert back to dictionary
    if len(cleaned_df) > 0:
        return cleaned_df.iloc[0].to_dict()
    else:
        return data  # Return original if cleaning failed


def validate_data_requirements(data: pd.DataFrame, model_type: str) -> Tuple[bool, List[str]]:
    """
    Validate that data meets minimum requirements for the specified model
    
    Args:
        data: DataFrame to validate
        model_type: Type of ML model
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check basic requirements
    if data.empty:
        issues.append("Dataset is empty")
        return False, issues
    
    if len(data) < 1:
        issues.append("Dataset must have at least 1 row")
        return False, issues
    
    # Model-specific validation
    required_columns = {
        'lead_score': ['company_size', 'budget', 'industry_score', 'engagement_score'],
        'churn': ['customer_tenure', 'monthly_charges', 'total_charges'],
        'sales_forecast': ['date', 'sales_amount'],
        'sentiment': ['text'],
        'keywords': ['text']
    }
    
    if model_type in required_columns:
        available_columns = set(data.columns.str.lower())
        required_cols = set([col.lower() for col in required_columns[model_type]])
        
        # Check if at least one required column exists
        if not any(req_col in available_columns for req_col in required_cols):
            issues.append(f"No required columns found for {model_type} model. Need at least one of: {required_columns[model_type]}")
    
    # Check data quality thresholds
    missing_percentage = (data.isnull().sum().sum() / data.size) * 100
    if missing_percentage > 90:
        issues.append(f"Too many missing values ({missing_percentage:.1f}%). Maximum allowed: 90%")
    
    return len(issues) == 0, issues