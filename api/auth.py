from flask import Blueprint, request, jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash
import jwt
import datetime
import uuid
from app import app, db
import logging

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def api_register():
    """API endpoint for user registration"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validation
        if not all([username, email, password]):
            return jsonify({'error': 'Username, email, and password are required'}), 400
        
        # Check if user exists
        from models import User, ApiKey
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'error': 'Username already exists'}), 409
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return jsonify({'error': 'Email already registered'}), 409
        
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
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': new_user.id,
            'username': new_user.username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'User created successfully',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email
            },
            'token': token,
            'api_key': api_key.key
        }), 201
        
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@auth_bp.route('/login', methods=['POST'])
def api_login():
    """API endpoint for user login"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'error': 'Username and password are required'}), 400
        
        from models import User, ApiKey
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.secret_key, algorithm='HS256')
        
        # Get user's API keys
        api_keys = ApiKey.query.filter_by(user_id=user.id, is_active=True).all()
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            },
            'token': token,
            'api_keys': [{
                'id': key.id,
                'key': key.key,
                'name': key.name,
                'created_at': key.created_at.isoformat(),
                'last_used': key.last_used.isoformat() if key.last_used else None
            } for key in api_keys]
        }), 200
        
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@auth_bp.route('/me', methods=['GET'])
def get_current_user():
    """Get current user info from JWT token"""
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    
    try:
        if token.startswith('Bearer '):
            token = token[7:]
        
        data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        user_id = data['user_id']
        
        from models import User, ApiKey
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        api_keys = ApiKey.query.filter_by(user_id=user.id, is_active=True).all()
        
        return jsonify({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat(),
                'is_active': user.is_active
            },
            'api_keys': [{
                'id': key.id,
                'key': key.key,
                'name': key.name,
                'created_at': key.created_at.isoformat(),
                'last_used': key.last_used.isoformat() if key.last_used else None,
                'rate_limit': key.rate_limit
            } for key in api_keys]
        }), 200
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        logging.error(f"Get user error: {str(e)}")
        return jsonify({'error': 'Failed to get user info'}), 500

@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """Refresh JWT token"""
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    
    try:
        if token.startswith('Bearer '):
            token = token[7:]
        
        # Decode token without verifying expiration
        data = jwt.decode(token, app.secret_key, algorithms=['HS256'], options={"verify_exp": False})
        user_id = data['user_id']
        username = data['username']
        
        from models import User
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({'error': 'User not found or inactive'}), 404
        
        # Generate new token
        new_token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Token refreshed successfully',
            'token': new_token
        }), 200
        
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        logging.error(f"Token refresh error: {str(e)}")
        return jsonify({'error': 'Failed to refresh token'}), 500

@auth_bp.route('/api-key', methods=['POST'])
def create_api_key():
    """Create a new API key for authenticated user"""
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    
    try:
        if token.startswith('Bearer '):
            token = token[7:]
        
        data_jwt = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        user_id = data_jwt['user_id']
        
        data = request.get_json()
        name = data.get('name', 'New API Key') if data else 'New API Key'
        
        from models import ApiKey
        new_key = ApiKey(
            user_id=user_id,
            key=str(uuid.uuid4()),
            name=name
        )
        db.session.add(new_key)
        db.session.commit()
        
        return jsonify({
            'message': 'API key created successfully',
            'api_key': {
                'id': new_key.id,
                'key': new_key.key,
                'name': new_key.name,
                'created_at': new_key.created_at.isoformat(),
                'rate_limit': new_key.rate_limit
            }
        }), 201
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        logging.error(f"API key creation error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create API key'}), 500

@auth_bp.route('/validate-key/<api_key>', methods=['GET'])
def validate_api_key(api_key):
    """Validate an API key"""
    try:
        from models import ApiKey, User
        key_obj = ApiKey.query.filter_by(key=api_key, is_active=True).first()
        
        if not key_obj:
            return jsonify({'valid': False, 'error': 'Invalid API key'}), 404
        
        user = User.query.get(key_obj.user_id)
        if not user or not user.is_active:
            return jsonify({'valid': False, 'error': 'User not found or inactive'}), 404
        
        return jsonify({
            'valid': True,
            'key_info': {
                'id': key_obj.id,
                'name': key_obj.name,
                'created_at': key_obj.created_at.isoformat(),
                'last_used': key_obj.last_used.isoformat() if key_obj.last_used else None,
                'rate_limit': key_obj.rate_limit
            },
            'user_info': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        }), 200
        
    except Exception as e:
        logging.error(f"API key validation error: {str(e)}")
        return jsonify({'valid': False, 'error': 'Validation failed'}), 500