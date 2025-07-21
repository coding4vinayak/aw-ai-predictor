"""
Model Manager for Dynamic Model Loading
Handles loading models from the Model Gallery and switching between versions
"""
import logging
import joblib
import os
from models_gallery import ModelGallery
from .utils import create_dummy_model

class ModelManager:
    """Manages model loading and selection based on Model Gallery"""
    
    def __init__(self):
        self.loaded_models = {}
        self.loaded_scalers = {}
    
    def get_active_model(self, model_type):
        """
        Get the active model for a given type from the gallery
        
        Args:
            model_type (str): Type of model (lead_scoring, churn_prediction, etc.)
            
        Returns:
            tuple: (model, scaler) objects
        """
        try:
            # Check if we have a cached model
            cache_key = f"{model_type}_active"
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key], self.loaded_scalers.get(cache_key)
            
            # Get active model from database
            active_model = ModelGallery.query.filter_by(
                model_type=model_type,
                is_active=True,
                is_default=True
            ).first()
            
            if not active_model:
                # Fallback to any active model of this type
                active_model = ModelGallery.query.filter_by(
                    model_type=model_type,
                    is_active=True
                ).first()
            
            if active_model and active_model.model_file_path:
                # Load model from file
                if os.path.exists(active_model.model_file_path):
                    model = joblib.load(active_model.model_file_path)
                    scaler = None
                    
                    # Load scaler if available
                    if active_model.scaler_file_path and os.path.exists(active_model.scaler_file_path):
                        scaler = joblib.load(active_model.scaler_file_path)
                    
                    # Cache the loaded model
                    self.loaded_models[cache_key] = model
                    if scaler:
                        self.loaded_scalers[cache_key] = scaler
                    
                    logging.info(f"Loaded model: {active_model.model_name} v{active_model.version}")
                    return model, scaler
                else:
                    logging.warning(f"Model file not found: {active_model.model_file_path}")
            
            # Fallback to dummy model if no active model found
            logging.info(f"No active model found for {model_type}, creating dummy model")
            model, scaler = create_dummy_model(model_type)
            return model, scaler
            
        except Exception as e:
            logging.error(f"Error loading model for {model_type}: {str(e)}")
            # Fallback to dummy model on error
            model, scaler = create_dummy_model(model_type)
            return model, scaler
    
    def get_model_by_id(self, model_id):
        """
        Get a specific model by ID from the gallery
        
        Args:
            model_id (int): ID of the model in the gallery
            
        Returns:
            tuple: (model, scaler, model_info) objects
        """
        try:
            model_info = ModelGallery.query.get(model_id)
            if not model_info:
                raise ValueError(f"Model with ID {model_id} not found")
            
            cache_key = f"model_{model_id}"
            
            # Check cache first
            if cache_key in self.loaded_models:
                return (self.loaded_models[cache_key], 
                       self.loaded_scalers.get(cache_key),
                       model_info)
            
            # Load from file
            if model_info.model_file_path and os.path.exists(model_info.model_file_path):
                model = joblib.load(model_info.model_file_path)
                scaler = None
                
                if model_info.scaler_file_path and os.path.exists(model_info.scaler_file_path):
                    scaler = joblib.load(model_info.scaler_file_path)
                
                # Cache the model
                self.loaded_models[cache_key] = model
                if scaler:
                    self.loaded_scalers[cache_key] = scaler
                
                return model, scaler, model_info
            else:
                raise FileNotFoundError(f"Model file not found: {model_info.model_file_path}")
                
        except Exception as e:
            logging.error(f"Error loading model {model_id}: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear the model cache"""
        self.loaded_models.clear()
        self.loaded_scalers.clear()
        logging.info("Model cache cleared")
    
    def get_all_models_info(self, model_type=None):
        """
        Get information about all models in the gallery
        
        Args:
            model_type (str, optional): Filter by model type
            
        Returns:
            list: List of model information dictionaries
        """
        try:
            query = ModelGallery.query
            if model_type:
                query = query.filter_by(model_type=model_type)
            
            models = query.order_by(ModelGallery.created_at.desc()).all()
            return [model.to_dict() for model in models]
            
        except Exception as e:
            logging.error(f"Error getting models info: {str(e)}")
            return []
    
    def update_model_stats(self, model_id, processing_time, success, confidence=None):
        """
        Update usage statistics for a model
        
        Args:
            model_id (int): ID of the model
            processing_time (float): Processing time in seconds
            success (bool): Whether the prediction was successful
            confidence (float, optional): Confidence score of the prediction
        """
        try:
            from api.model_gallery import model_gallery_bp
            from models_gallery import ModelUsageStats
            from datetime import datetime
            from app import db
            
            today = datetime.utcnow().date()
            
            # Get or create stats entry
            stats = ModelUsageStats.query.filter_by(
                model_id=model_id,
                date=today
            ).first()
            
            if not stats:
                stats = ModelUsageStats(
                    model_id=model_id,
                    date=today,
                    usage_count=0,
                    total_processing_time=0.0,
                    success_rate=0.0,
                    average_confidence=0.0
                )
                db.session.add(stats)
            
            # Update statistics
            stats.usage_count += 1
            stats.total_processing_time += processing_time
            
            # Update success rate (moving average)
            current_rate = stats.success_rate
            stats.success_rate = (current_rate * (stats.usage_count - 1) + (1 if success else 0)) / stats.usage_count
            
            # Update average confidence (moving average)
            if confidence is not None:
                current_avg = stats.average_confidence
                stats.average_confidence = (current_avg * (stats.usage_count - 1) + confidence) / stats.usage_count
            
            db.session.commit()
            logging.info(f"Updated stats for model {model_id}")
            
        except Exception as e:
            logging.error(f"Error updating model stats: {str(e)}")

# Global model manager instance
model_manager = ModelManager()