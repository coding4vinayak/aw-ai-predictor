from flask import Blueprint, request, jsonify, render_template
import os
import json
import joblib
from datetime import datetime
from werkzeug.utils import secure_filename
from app import db, api_key_required
from models_gallery import ModelGallery, ModelUsageStats
from models import User

model_gallery_bp = Blueprint('model_gallery', __name__)

# Allowed file extensions for model uploads
ALLOWED_EXTENSIONS = {'pkl', 'joblib', 'model'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@model_gallery_bp.route('/models', methods=['GET'])
def get_all_models():
    """Get all models in the gallery"""
    try:
        model_type = request.args.get('model_type')
        status = request.args.get('status')
        is_active = request.args.get('is_active')
        
        query = ModelGallery.query
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        if status:
            query = query.filter_by(status=status)
        if is_active is not None:
            query = query.filter_by(is_active=is_active.lower() == 'true')
            
        models = query.order_by(ModelGallery.created_at.desc()).all()
        
        return jsonify({
            'models': [model.to_dict() for model in models],
            'total_count': len(models)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models/<int:model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get detailed information about a specific model"""
    try:
        model = ModelGallery.query.get_or_404(model_id)
        
        # Get usage statistics
        usage_stats = ModelUsageStats.query.filter_by(model_id=model_id).all()
        
        model_dict = model.to_dict()
        model_dict['usage_statistics'] = [
            {
                'date': stat.date.isoformat(),
                'usage_count': stat.usage_count,
                'total_processing_time': stat.total_processing_time,
                'success_rate': stat.success_rate,
                'average_confidence': stat.average_confidence
            }
            for stat in usage_stats
        ]
        
        return jsonify(model_dict)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models', methods=['POST'])
@api_key_required
def add_model(user_id):
    """Add a new model to the gallery"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['model_name', 'model_type', 'version']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        # Check if model with same name and version exists
        existing_model = ModelGallery.query.filter_by(
            model_name=data['model_name'],
            version=data['version']
        ).first()
        
        if existing_model:
            return jsonify({'error': 'Model with this name and version already exists'}), 409
        
        # Create new model entry
        model = ModelGallery(
            model_name=data['model_name'],
            model_type=data['model_type'],
            version=data['version'],
            description=data.get('description', ''),
            model_file_path=data.get('model_file_path'),
            scaler_file_path=data.get('scaler_file_path'),
            accuracy=data.get('accuracy'),
            precision=data.get('precision'),
            recall=data.get('recall'),
            f1_score=data.get('f1_score'),
            training_data_size=data.get('training_data_size'),
            is_active=data.get('is_active', False),
            is_default=data.get('is_default', False),
            status=data.get('status', 'trained'),
            hyperparameters=json.dumps(data.get('hyperparameters', {})),
            features=json.dumps(data.get('features', [])),
            is_custom=data.get('is_custom', True),
            uploaded_by=user_id
        )
        
        db.session.add(model)
        db.session.commit()
        
        return jsonify({
            'message': 'Model added successfully',
            'model': model.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models/<int:model_id>', methods=['PUT'])
@api_key_required
def update_model(user_id, model_id):
    """Update model information"""
    try:
        model = ModelGallery.query.get_or_404(model_id)
        data = request.get_json()
        
        # Update fields if provided
        if 'description' in data:
            model.description = data['description']
        if 'is_active' in data:
            model.is_active = data['is_active']
        if 'is_default' in data:
            model.is_default = data['is_default']
        if 'status' in data:
            model.status = data['status']
        if 'hyperparameters' in data:
            model.hyperparameters = json.dumps(data['hyperparameters'])
        if 'features' in data:
            model.features = json.dumps(data['features'])
        
        model.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Model updated successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models/<int:model_id>/activate', methods=['POST'])
@api_key_required
def activate_model(user_id, model_id):
    """Activate a model and optionally set as default"""
    try:
        model = ModelGallery.query.get_or_404(model_id)
        data = request.get_json() or {}
        
        # Activate the model
        model.is_active = True
        model.status = 'deployed'
        
        # If setting as default, deactivate other defaults of same type
        if data.get('set_as_default', False):
            ModelGallery.query.filter_by(
                model_type=model.model_type,
                is_default=True
            ).update({'is_default': False})
            
            model.is_default = True
        
        model.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Model activated successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models/<int:model_id>/deactivate', methods=['POST'])
@api_key_required
def deactivate_model(user_id, model_id):
    """Deactivate a model"""
    try:
        model = ModelGallery.query.get_or_404(model_id)
        
        model.is_active = False
        model.is_default = False
        model.status = 'deprecated'
        model.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Model deactivated successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models/<int:model_id>/upload', methods=['POST'])
@api_key_required
def upload_model_files(user_id, model_id):
    """Upload model and scaler files"""
    try:
        model = ModelGallery.query.get_or_404(model_id)
        
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        model_file = request.files['model_file']
        scaler_file = request.files.get('scaler_file')
        
        if model_file and allowed_file(model_file.filename):
            # Create models directory if it doesn't exist
            models_dir = 'ml_models/custom'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model file
            model_filename = secure_filename(f"{model.model_name}_v{model.version}_model.pkl")
            model_path = os.path.join(models_dir, model_filename)
            model_file.save(model_path)
            model.model_file_path = model_path
            
            # Save scaler file if provided
            if scaler_file and allowed_file(scaler_file.filename):
                scaler_filename = secure_filename(f"{model.model_name}_v{model.version}_scaler.pkl")
                scaler_path = os.path.join(models_dir, scaler_filename)
                scaler_file.save(scaler_path)
                model.scaler_file_path = scaler_path
            
            model.updated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'message': 'Model files uploaded successfully',
                'model_path': model_path,
                'scaler_path': scaler_path if scaler_file else None
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/model-types', methods=['GET'])
def get_model_types():
    """Get all available model types"""
    try:
        model_types = db.session.query(ModelGallery.model_type).distinct().all()
        types_list = [mt[0] for mt in model_types]
        
        # Add predefined types if not in database
        predefined_types = [
            'lead_scoring', 'churn_prediction', 'sales_forecast', 'nlp_analysis',
            'healthcare_risk', 'financial_fraud', 'retail_optimization', 
            'manufacturing_quality', 'education_performance', 'real_estate_valuation'
        ]
        
        all_types = list(set(types_list + predefined_types))
        
        return jsonify({
            'model_types': sorted(all_types),
            'count': len(all_types)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_gallery_bp.route('/models/<int:model_id>/stats', methods=['POST'])
@api_key_required
def update_model_stats(user_id, model_id):
    """Update model usage statistics"""
    try:
        data = request.get_json()
        today = datetime.utcnow().date()
        
        # Get or create today's stats
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
        if 'processing_time' in data:
            stats.total_processing_time += data['processing_time']
        if 'success' in data:
            # Update success rate (simple moving average)
            current_rate = stats.success_rate
            stats.success_rate = (current_rate * (stats.usage_count - 1) + (1 if data['success'] else 0)) / stats.usage_count
        if 'confidence' in data:
            # Update average confidence (simple moving average)
            current_avg = stats.average_confidence
            stats.average_confidence = (current_avg * (stats.usage_count - 1) + data['confidence']) / stats.usage_count
        
        db.session.commit()
        
        return jsonify({'message': 'Statistics updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500