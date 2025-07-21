from flask import Blueprint, request, jsonify, send_file
import os
import pandas as pd
import json
import uuid
import numpy as np
from werkzeug.utils import secure_filename
from app import db, app, api_key_required
from models import FileUpload, PredictionLog
from ml_services.data_cleaner import DataCleaner, validate_data_requirements, clean_api_data
from api.predictions import convert_numpy_types
import time

uploads_bp = Blueprint('uploads', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@uploads_bp.route('/', methods=['POST'])
@api_key_required
def upload_file(user_id):
    """Upload CSV file for batch predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_type = request.form.get('model_type')
        
        if not model_type:
            return jsonify({'error': 'Model type is required'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use CSV, XLS, or XLSX'}), 400
        
        # Generate unique filename with safety check
        if not file.filename:
            return jsonify({'error': 'No filename provided'}), 400
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        
        # Read and validate the file
        try:
            if original_filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            total_rows = len(df)
            
            # Validate data requirements for the specified model
            is_valid, issues = validate_data_requirements(df, model_type)
            if not is_valid:
                os.remove(filepath)
                return jsonify({
                    'error': 'Data validation failed',
                    'details': issues,
                    'recommendation': 'Please check the Data Guide for required columns and formats'
                }), 400
            
            # Perform initial data quality assessment
            cleaner = DataCleaner()
            cleaned_data, cleaning_report = cleaner.clean_data(df.copy(), model_type)
            
            # Store cleaning report for later reference
            quality_score = cleaning_report.get('data_quality_score', 0.5)
            
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Create file upload record
        file_upload = FileUpload(
            user_id=user_id,
            filename=unique_filename,
            original_filename=original_filename,
            file_size=file_size,
            model_type=model_type,
            total_rows=total_rows,
            status='uploaded'
        )
        db.session.add(file_upload)
        db.session.commit()
        
        return jsonify({
            'message': 'File uploaded successfully',
            'upload_id': int(file_upload.id),
            'filename': original_filename,
            'total_rows': int(total_rows),
            'model_type': model_type,
            'data_quality_score': float(quality_score),
            'cleaning_summary': {
                'missing_values_filled': int(cleaning_report.get('missing_values_filled', 0)),
                'formats_standardized': int(cleaning_report.get('formats_standardized', 0)),
                'outliers_handled': int(cleaning_report.get('outliers_handled', 0)),
                'duplicates_removed': int(cleaning_report.get('duplicates_removed', 0))
            },
            'warnings': list(cleaning_report.get('warnings', []))
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@uploads_bp.route('/<int:upload_id>/process', methods=['POST'])
@api_key_required
def process_file(user_id, upload_id):
    """Process uploaded file and generate predictions"""
    try:
        file_upload = FileUpload.query.filter_by(id=upload_id, user_id=user_id).first()
        if not file_upload:
            return jsonify({'error': 'File upload not found'}), 404
        
        if file_upload.status != 'uploaded':
            return jsonify({'error': 'File already processed or processing'}), 400
        
        # Update status to processing
        file_upload.status = 'processing'
        db.session.commit()
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_upload.filename)
        
        # Read and clean the file
        if file_upload.original_filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Clean the data before processing
        cleaner = DataCleaner()
        df_cleaned, cleaning_report = cleaner.clean_data(df, file_upload.model_type)
        
        # Process predictions based on model type
        results = []
        processed_count = 0
        
        for index, row in df_cleaned.iterrows():
            try:
                start_time = time.time()
                
                # Convert row to dict for prediction (handle numpy types)
                row_data = convert_numpy_types(row.to_dict())
                
                # Additional cleaning for individual row
                row_data = clean_api_data(row_data, file_upload.model_type)
                
                # Get prediction based on model type
                if file_upload.model_type == 'lead_score':
                    from ml_services.lead_scoring import predict_lead_score
                    prediction = predict_lead_score(row_data)
                elif file_upload.model_type == 'churn':
                    from ml_services.churn_prediction import predict_churn
                    prediction = predict_churn(row_data)
                elif file_upload.model_type == 'sales_forecast':
                    from ml_services.sales_forecast import predict_sales
                    prediction = predict_sales(row_data)
                else:
                    raise ValueError(f"Unsupported model type: {file_upload.model_type}")
                
                processing_time = time.time() - start_time
                
                # Convert numpy types and add prediction to results
                prediction_converted = convert_numpy_types(prediction)
                result_row = convert_numpy_types(row_data.copy())
                result_row.update(prediction_converted)
                results.append(result_row)
                
                # Log individual prediction
                log = PredictionLog(
                    user_id=user_id,
                    model_type=file_upload.model_type,
                    input_data=json.dumps(row_data),
                    prediction=json.dumps(prediction_converted),
                    confidence=float(prediction_converted.get('confidence', 0.0)),
                    processing_time=float(processing_time),
                    status='success'
                )
                db.session.add(log)
                
                processed_count += 1
                
                # Update progress every 10 rows
                if processed_count % 10 == 0:
                    file_upload.processed_rows = processed_count
                    db.session.commit()
                
            except Exception as e:
                # Log error for this row
                log = PredictionLog(
                    user_id=user_id,
                    model_type=file_upload.model_type,
                    input_data=json.dumps(row_data if 'row_data' in locals() else {}),
                    status='error',
                    error_message=str(e),
                    processing_time=float(0)
                )
                db.session.add(log)
                
                # Add error to results (convert numpy types)
                result_row = convert_numpy_types(row.to_dict())
                result_row['error'] = str(e)
                results.append(result_row)
                
        # Save results to file
        results_df = pd.DataFrame(results)
        results_filename = f"results_{uuid.uuid4()}.csv"
        results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
        results_df.to_csv(results_filepath, index=False)
        
        # Update file upload record
        file_upload.status = 'completed'
        file_upload.processed_rows = len(results)
        file_upload.results_file = results_filename
        file_upload.completed_at = pd.Timestamp.now()
        db.session.commit()
        
        return jsonify({
            'message': 'File processed successfully',
            'processed_rows': len(results),
            'results_file': results_filename,
            'download_url': f'/api/upload/{upload_id}/download'
        })
        
    except Exception as e:
        # Update status to error
        if 'file_upload' in locals():
            file_upload.status = 'error'
            file_upload.error_message = str(e)
            db.session.commit()
        
        return jsonify({'error': str(e)}), 500

@uploads_bp.route('/<int:upload_id>/download', methods=['GET'])
@api_key_required
def download_results(user_id, upload_id):
    """Download processed results"""
    try:
        file_upload = FileUpload.query.filter_by(id=upload_id, user_id=user_id).first()
        if not file_upload:
            return jsonify({'error': 'File upload not found'}), 404
        
        if file_upload.status != 'completed' or not file_upload.results_file:
            return jsonify({'error': 'Results not available'}), 400
        
        results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_upload.results_file)
        
        if not os.path.exists(results_filepath):
            return jsonify({'error': 'Results file not found'}), 404
        
        return send_file(
            results_filepath,
            as_attachment=True,
            download_name=f"results_{file_upload.original_filename}"
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@uploads_bp.route('/', methods=['GET'])
@api_key_required
def get_uploads(user_id):
    """Get user's file uploads"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        uploads = FileUpload.query.filter_by(user_id=user_id).order_by(
            FileUpload.created_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'uploads': [{
                'id': upload.id,
                'original_filename': upload.original_filename,
                'model_type': upload.model_type,
                'status': upload.status,
                'total_rows': upload.total_rows,
                'processed_rows': upload.processed_rows,
                'created_at': upload.created_at.isoformat(),
                'completed_at': upload.completed_at.isoformat() if upload.completed_at else None,
                'error_message': upload.error_message,
                'has_results': bool(upload.results_file)
            } for upload in uploads.items],
            'pagination': {
                'page': uploads.page,
                'per_page': uploads.per_page,
                'total': uploads.total,
                'pages': uploads.pages
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@uploads_bp.route('/<int:upload_id>/status', methods=['GET'])
@api_key_required
def get_upload_status(user_id, upload_id):
    """Get upload processing status"""
    try:
        file_upload = FileUpload.query.filter_by(id=upload_id, user_id=user_id).first()
        if not file_upload:
            return jsonify({'error': 'File upload not found'}), 404
        
        progress = 0
        if file_upload.total_rows > 0:
            progress = (file_upload.processed_rows / file_upload.total_rows) * 100
        
        return jsonify({
            'status': file_upload.status,
            'progress': round(progress, 2),
            'processed_rows': file_upload.processed_rows,
            'total_rows': file_upload.total_rows,
            'error_message': file_upload.error_message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
