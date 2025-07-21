import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime, timedelta
import jwt
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure CORS
CORS(app)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///ai_platform.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the app with the extension
db.init_app(app)

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user_id, *args, **kwargs)
    return decorated

# Flexible authentication decorator (supports both API key and JWT token)
def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Try API key first
        api_key = request.headers.get('X-API-Key')
        if api_key:
            from models import ApiKey
            key_obj = ApiKey.query.filter_by(key=api_key, is_active=True).first()
            if key_obj:
                # Update last used timestamp
                key_obj.last_used = datetime.utcnow()
                db.session.commit()
                return f(key_obj.user_id, *args, **kwargs)
        
        # Try JWT token
        token = request.headers.get('Authorization')
        if token:
            try:
                if token.startswith('Bearer '):
                    token = token[7:]
                data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
                current_user_id = data['user_id']
                return f(current_user_id, *args, **kwargs)
            except:
                pass
        
        return jsonify({'message': 'Authentication required! Provide X-API-Key header or Authorization: Bearer token'}), 401
    return decorated

# API key authentication decorator (backward compatibility)
def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'message': 'API key is missing!'}), 401
        
        from models import ApiKey
        key_obj = ApiKey.query.filter_by(key=api_key, is_active=True).first()
        if not key_obj:
            return jsonify({'message': 'Invalid API key!'}), 401
        
        # Update last used timestamp
        key_obj.last_used = datetime.utcnow()
        db.session.commit()
        
        return f(key_obj.user_id, *args, **kwargs)
    return decorated

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            return render_template('register.html', error='All fields are required')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        # Additional validation to ensure password is a string
        if not isinstance(password, str) or len(password.strip()) == 0:
            return render_template('register.html', error='Invalid password provided')
        
        # Check if user exists
        from models import User, ApiKey
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error='Username already exists')
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return render_template('register.html', error='Email already registered')
        
        # Create new user
        new_user = User()
        new_user.username = username
        new_user.email = email
        new_user.password_hash = generate_password_hash(password)
        db.session.add(new_user)
        db.session.commit()
        
        # Create default API key for new user
        api_key = ApiKey(
            user_id=new_user.id,
            key=str(uuid.uuid4()),
            name='Default API Key'
        )
        db.session.add(api_key)
        db.session.commit()
        
        return redirect(url_for('login', success='Account created successfully! You can now sign in.'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    success = request.args.get('success')
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return render_template('login.html', error='Username and password are required')
        
        from models import User
        user = User.query.filter_by(username=username).first()
        
        if user and user.password_hash and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html', success=success)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Routes
@app.route('/')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    from models import User, ApiKey
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('login'))
    api_keys = ApiKey.query.filter_by(user_id=user.id).all()
    
    return render_template('dashboard.html', user=user, api_keys=api_keys)

@app.route('/api-docs')
def api_docs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('api_docs.html')

@app.route('/getting-started')
def getting_started():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('getting_started.html')

@app.route('/api-tester')
def api_tester():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    from models import User, ApiKey
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('login'))
    api_keys = ApiKey.query.filter_by(user_id=user.id, is_active=True).all()
    
    return render_template('api_tester.html', user=user, api_keys=api_keys)

@app.route('/data-guide')
def data_guide():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('data_guide.html')

# API Key management routes
@app.route('/api/generate-key', methods=['POST'])
def generate_api_key():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    name = request.form.get('name', 'New API Key')
    
    from models import ApiKey
    new_key = ApiKey(
        user_id=session['user_id'],
        key=str(uuid.uuid4()),
        name=name
    )
    db.session.add(new_key)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'key': new_key.key,
        'name': new_key.name,
        'id': new_key.id
    })

@app.route('/api/delete-key/<int:key_id>', methods=['DELETE'])
def delete_api_key(key_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from models import ApiKey
    api_key = ApiKey.query.filter_by(id=key_id, user_id=session['user_id']).first()
    
    if not api_key:
        return jsonify({'error': 'API key not found'}), 404
    
    db.session.delete(api_key)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/toggle-key/<int:key_id>', methods=['POST'])
def toggle_api_key(key_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from models import ApiKey
    api_key = ApiKey.query.filter_by(id=key_id, user_id=session['user_id']).first()
    
    if not api_key:
        return jsonify({'error': 'API key not found'}), 404
    
    api_key.is_active = not api_key.is_active
    db.session.commit()
    
    return jsonify({
        'success': True,
        'is_active': api_key.is_active
    })

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

# Register blueprints
from api.auth import auth_bp
from api.predictions import predictions_bp
from api.uploads import uploads_bp
from api.admin import admin_bp
from api.industry_endpoints import industry_bp
from api.specialized_endpoints import specialized_bp
from api.model_gallery import model_gallery_bp
from api.credit_manager import CreditManager
from monitoring.dashboard import dashboard_bp
from api.enterprise_endpoints import enterprise_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(predictions_bp, url_prefix='/api/predict')
app.register_blueprint(uploads_bp, url_prefix='/api/upload')
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(industry_bp, url_prefix='/api/industry')
app.register_blueprint(specialized_bp, url_prefix='/api/specialized')
app.register_blueprint(model_gallery_bp, url_prefix='/api/model-gallery')
app.register_blueprint(dashboard_bp)
app.register_blueprint(enterprise_bp, url_prefix='/api/enterprise')

# CRM connector endpoints
@app.route('/api/connectors/hubspot/leads')
@api_key_required
def hubspot_leads(user_id):
    try:
        from connectors.hubspot import HubSpotConnector
        connector = HubSpotConnector()
        leads = connector.get_leads()
        return jsonify({'leads': leads, 'count': len(leads)})
    except Exception as e:
        logging.error(f"HubSpot connector error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/connectors/zoho/leads')
@api_key_required
def zoho_leads(user_id):
    try:
        from connectors.zoho import ZohoConnector
        connector = ZohoConnector()
        leads = connector.get_leads()
        return jsonify({'leads': leads, 'count': len(leads)})
    except Exception as e:
        logging.error(f"Zoho connector error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Model Gallery route
@app.route('/models')
def model_gallery():
    """Model Gallery page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('model_gallery.html')

# ML Service Pages
@app.route('/ml-services/lead-scoring')
def lead_scoring_page():
    """Lead Scoring Model page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('ml_services/lead_scoring.html')

@app.route('/ml-services/churn-prediction')
def churn_prediction_page():
    """Churn Prediction Model page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('ml_services/churn_prediction.html')

@app.route('/ml-services/sales-forecast')
def sales_forecast_page():
    """Sales Forecast Model page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('ml_services/sales_forecast.html')

# Initialize database
with app.app_context():
    import models
    import models_gallery  # Import the new model gallery models
    db.create_all()
    
    # Setup default data (creates plans and admin user)
    from api.admin import setup_default_data
    setup_default_data()
    
    # Create a default API key for admin if none exists
    admin_user = models.User.query.filter_by(username='admin').first()
    if admin_user and not admin_user.api_keys:
        default_key = models.ApiKey(
            user_id=admin_user.id,
            key='demo-api-key-12345',
            name='Default Demo Key'
        )
        db.session.add(default_key)
        db.session.commit()
        logging.info("Created default API key: demo-api-key-12345")
    
    # Add default models to the gallery if none exist
    from models_gallery import ModelGallery
    if not ModelGallery.query.first():
        default_models = [
            {
                'model_name': 'Lead Scoring Model',
                'model_type': 'lead_scoring',
                'version': '1.0.0',
                'description': 'Default lead scoring model using RandomForest classifier',
                'accuracy': 85.5,
                'precision': 82.3,
                'recall': 88.1,
                'f1_score': 85.1,
                'is_active': True,
                'is_default': True,
                'status': 'deployed',
                'features': '["company_size", "budget", "industry_score", "engagement_score", "demographic_score", "behavioral_score", "source_score"]'
            },
            {
                'model_name': 'Churn Prediction Model',
                'model_type': 'churn_prediction', 
                'version': '1.0.0',
                'description': 'Default churn prediction model using GradientBoosting classifier',
                'accuracy': 89.2,
                'precision': 87.4,
                'recall': 91.0,
                'f1_score': 89.2,
                'is_active': True,
                'is_default': True,
                'status': 'deployed',
                'features': '["tenure", "monthly_charges", "total_charges", "contract_type", "payment_method", "internet_service", "support_tickets"]'
            },
            {
                'model_name': 'Sales Forecast Model',
                'model_type': 'sales_forecast',
                'version': '1.0.0', 
                'description': 'Default sales forecasting model using RandomForest regressor',
                'accuracy': 92.1,
                'precision': 90.8,
                'recall': 93.5,
                'f1_score': 92.1,
                'is_active': True,
                'is_default': True,
                'status': 'deployed',
                'features': '["historical_sales", "seasonality", "marketing_spend", "economic_indicators", "product_category"]'
            }
        ]
        
        for model_data in default_models:
            model = ModelGallery(**model_data)
            db.session.add(model)
        
        db.session.commit()
        logging.info("Created default model gallery entries")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
