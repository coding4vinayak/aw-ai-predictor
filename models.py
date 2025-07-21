from app import db
from datetime import datetime
import uuid

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    api_keys = db.relationship('ApiKey', backref='user', lazy=True)
    prediction_logs = db.relationship('PredictionLog', backref='user', lazy=True)
    user_credits = db.relationship('UserCredit', backref='user', lazy=True)
    usage_logs = db.relationship('UsageLog', backref='user', lazy=True)

class ApiKey(db.Model):
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    key = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    rate_limit = db.Column(db.Integer, default=1000)  # requests per hour
    
    # Relationships
    usage_logs = db.relationship('UsageLog', backref='api_key', lazy=True)
    
    def __init__(self, **kwargs):
        super(ApiKey, self).__init__(**kwargs)
        if not self.key:
            self.key = str(uuid.uuid4())

class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # lead_score, churn, etc.
    input_data = db.Column(db.Text)  # JSON string of input
    prediction = db.Column(db.Text)  # JSON string of prediction
    confidence = db.Column(db.Float)
    processing_time = db.Column(db.Float)  # in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='success')  # success, error
    error_message = db.Column(db.Text)

class FileUpload(db.Model):
    __tablename__ = 'file_uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    model_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='processing')  # processing, completed, error
    results_file = db.Column(db.String(255))  # path to results file
    total_rows = db.Column(db.Integer)
    processed_rows = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)

class CreditPlan(db.Model):
    __tablename__ = 'credit_plans'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    credits_per_month = db.Column(db.Integer, nullable=False)
    max_file_size_mb = db.Column(db.Integer, default=16)
    max_requests_per_hour = db.Column(db.Integer, default=100)
    price_per_month = db.Column(db.Float, default=0.0)
    features = db.Column(db.Text)  # JSON string of features list
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user_credits = db.relationship('UserCredit', backref='plan', lazy=True)

class UserCredit(db.Model):
    __tablename__ = 'user_credits'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey('credit_plans.id'), nullable=False)
    credits_used = db.Column(db.Integer, default=0)
    credits_remaining = db.Column(db.Integer, nullable=False)
    month_year = db.Column(db.String(7), nullable=False)  # Format: YYYY-MM
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint for user per month
    __table_args__ = (db.UniqueConstraint('user_id', 'month_year', name='uq_user_month'),)

class UsageLog(db.Model):
    __tablename__ = 'usage_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_keys.id'))
    endpoint = db.Column(db.String(100), nullable=False)
    method = db.Column(db.String(10), nullable=False)
    credits_used = db.Column(db.Integer, default=0)
    data_size_kb = db.Column(db.Float, default=0.0)
    processing_time = db.Column(db.Float)
    status_code = db.Column(db.Integer, nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class IndustryModel(db.Model):
    __tablename__ = 'industry_models'
    
    id = db.Column(db.Integer, primary_key=True)
    industry_name = db.Column(db.String(50), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    model_version = db.Column(db.String(20), default='v1.0')
    model_path = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, default=True)
    accuracy_score = db.Column(db.Float)
    required_features = db.Column(db.Text)  # JSON string of required features
    feature_weights = db.Column(db.Text)    # JSON string of feature weights
    preprocessing_config = db.Column(db.Text)  # JSON string of preprocessing config
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint for industry-model combination
    __table_args__ = (db.UniqueConstraint('industry_name', 'model_type', name='uq_industry_model'),)