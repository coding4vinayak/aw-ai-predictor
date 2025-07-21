from app import db
from datetime import datetime
import json

class ModelGallery(db.Model):
    __tablename__ = 'model_gallery'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'lead_scoring', 'churn', 'sales_forecast', etc.
    version = db.Column(db.String(20), nullable=False, default='1.0.0')
    description = db.Column(db.Text)
    model_file_path = db.Column(db.String(255))
    scaler_file_path = db.Column(db.String(255))
    
    # Model metadata
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    training_data_size = db.Column(db.Integer)
    
    # Model status
    is_active = db.Column(db.Boolean, default=False)
    is_default = db.Column(db.Boolean, default=False)
    status = db.Column(db.String(20), default='trained')  # 'training', 'trained', 'deployed', 'deprecated'
    
    # Model configuration
    hyperparameters = db.Column(db.Text)  # JSON string of model parameters
    features = db.Column(db.Text)  # JSON string of required features
    
    # Custom model info
    is_custom = db.Column(db.Boolean, default=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    uploader = db.relationship('User', backref='uploaded_models', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'description': self.description,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'training_data_size': self.training_data_size,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'status': self.status,
            'is_custom': self.is_custom,
            'hyperparameters': json.loads(self.hyperparameters) if self.hyperparameters else {},
            'features': json.loads(self.features) if self.features else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ModelUsageStats(db.Model):
    __tablename__ = 'model_usage_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model_gallery.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    usage_count = db.Column(db.Integer, default=0)
    total_processing_time = db.Column(db.Float, default=0.0)
    success_rate = db.Column(db.Float, default=0.0)
    average_confidence = db.Column(db.Float, default=0.0)
    
    # Relationships
    model = db.relationship('ModelGallery', backref='usage_stats', lazy=True)