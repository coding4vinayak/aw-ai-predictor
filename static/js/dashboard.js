// Dashboard JavaScript
class DashboardManager {
    constructor() {
        this.apiKey = 'demo-api-key-12345';
        this.currentModel = 'lead_score';
        this.charts = {};
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.checkApiHealth();
        this.loadPredictionLogs();
        this.loadUploadHistory();
        this.initializeCharts();
    }
    
    bindEvents() {
        // Model selection
        document.querySelectorAll('.model-list .list-group-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.selectModel(e.currentTarget);
            });
        });
        
        // Navbar model links
        document.querySelectorAll('[data-model]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const modelType = e.currentTarget.dataset.model;
                this.selectModelByType(modelType);
                
                // Switch to test tab
                const testTab = document.getElementById('nav-test-tab');
                if (testTab) {
                    testTab.click();
                }
            });
        });
        
        // Prediction form
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm) {
            predictionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.makePrediction();
            });
        }
        
        // Upload form
        const uploadForm = document.getElementById('uploadForm');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.uploadFile();
            });
        }
        
        // CRM connectors
        document.getElementById('loadHubspotLeads')?.addEventListener('click', () => {
            this.loadCrmData('hubspot');
        });
        
        document.getElementById('loadZohoLeads')?.addEventListener('click', () => {
            this.loadCrmData('zoho');
        });
        
        // Refresh logs
        document.getElementById('refreshLogs')?.addEventListener('click', () => {
            this.loadPredictionLogs();
        });
        
        // API Key copy
        document.getElementById('copyApiKey')?.addEventListener('click', () => {
            this.copyApiKey();
        });
        
        // Tab switches
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const targetId = e.target.getAttribute('data-bs-target');
                if (targetId === '#nav-logs') {
                    this.loadPredictionLogs();
                } else if (targetId === '#nav-upload') {
                    this.loadUploadHistory();
                }
            });
        });
    }
    
    selectModel(element) {
        // Update active state
        document.querySelectorAll('.model-list .list-group-item').forEach(item => {
            item.classList.remove('active');
        });
        element.classList.add('active');
        
        // Get model type
        this.currentModel = element.dataset.model;
        
        // Update input form
        this.updateInputForm();
    }
    
    selectModelByType(modelType) {
        const modelElement = document.querySelector(`[data-model="${modelType}"]`);
        if (modelElement && modelElement.closest('.model-list')) {
            this.selectModel(modelElement);
        }
    }
    
    updateInputForm() {
        const inputFields = document.getElementById('inputFields');
        if (!inputFields) return;
        
        const fieldConfigs = {
            lead_score: [
                { name: 'company_size', label: 'Company Size', type: 'number', placeholder: '100', help: 'Number of employees' },
                { name: 'budget', label: 'Budget', type: 'number', placeholder: '10000', help: 'Annual budget in USD' },
                { name: 'industry_score', label: 'Industry Score', type: 'number', placeholder: '4', help: 'Industry relevance (1-5)' },
                { name: 'engagement_score', label: 'Engagement Score', type: 'number', placeholder: '3', help: 'Engagement level (1-5)' },
                { name: 'demographic_score', label: 'Demographic Score', type: 'number', placeholder: '4', help: 'Demographic fit (1-5)' },
                { name: 'behavioral_score', label: 'Behavioral Score', type: 'number', placeholder: '3', help: 'Behavioral indicators (1-5)' },
                { name: 'source_score', label: 'Source Score', type: 'number', placeholder: '4', help: 'Lead source quality (1-5)' }
            ],
            churn: [
                { name: 'tenure_months', label: 'Tenure (Months)', type: 'number', placeholder: '24', help: 'Customer tenure in months' },
                { name: 'monthly_charges', label: 'Monthly Charges', type: 'number', placeholder: '75', help: 'Monthly charges in USD' },
                { name: 'total_charges', label: 'Total Charges', type: 'number', placeholder: '1800', help: 'Total charges to date' },
                { name: 'contract_length', label: 'Contract Length', type: 'number', placeholder: '12', help: 'Contract length in months' },
                { name: 'payment_method_score', label: 'Payment Method Score', type: 'number', placeholder: '3', help: 'Payment reliability (1-3)' },
                { name: 'support_calls', label: 'Support Calls', type: 'number', placeholder: '2', help: 'Support calls last month' },
                { name: 'usage_score', label: 'Usage Score', type: 'number', placeholder: '4', help: 'Product usage level (1-5)' },
                { name: 'satisfaction_score', label: 'Satisfaction Score', type: 'number', placeholder: '4', help: 'Customer satisfaction (1-5)' }
            ],
            sales_forecast: [
                { name: 'historical_sales_avg', label: 'Historical Sales Average', type: 'number', placeholder: '50000', help: 'Average monthly sales' },
                { name: 'seasonality_factor', label: 'Seasonality Factor', type: 'number', placeholder: '1.2', help: 'Seasonal multiplier' },
                { name: 'market_trend', label: 'Market Trend', type: 'number', placeholder: '1.05', help: 'Market growth factor' },
                { name: 'marketing_spend', label: 'Marketing Spend', type: 'number', placeholder: '5000', help: 'Monthly marketing spend' },
                { name: 'lead_volume', label: 'Lead Volume', type: 'number', placeholder: '200', help: 'Monthly lead volume' },
                { name: 'conversion_rate', label: 'Conversion Rate', type: 'number', placeholder: '0.15', help: 'Lead to customer rate' },
                { name: 'economic_indicator', label: 'Economic Indicator', type: 'number', placeholder: '1.0', help: 'Economic health factor' },
                { name: 'competitor_activity', label: 'Competitor Activity', type: 'number', placeholder: '0.5', help: 'Competitive pressure (0-1)' }
            ],
            sentiment: [
                { name: 'text', label: 'Text to Analyze', type: 'textarea', placeholder: 'Enter text for sentiment analysis...', help: 'Any text content' }
            ],
            keywords: [
                { name: 'text', label: 'Text to Process', type: 'textarea', placeholder: 'Enter text for keyword extraction...', help: 'Any text content' }
            ]
        };
        
        const fields = fieldConfigs[this.currentModel] || [];
        
        inputFields.innerHTML = fields.map(field => {
            if (field.type === 'textarea') {
                return `
                    <div class="mb-3">
                        <label for="${field.name}" class="form-label">${field.label}</label>
                        <textarea class="form-control" id="${field.name}" name="${field.name}" 
                                  placeholder="${field.placeholder}" rows="4" required></textarea>
                        <div class="form-text">${field.help}</div>
                    </div>
                `;
            } else {
                return `
                    <div class="mb-3">
                        <label for="${field.name}" class="form-label">${field.label}</label>
                        <input type="${field.type}" class="form-control" id="${field.name}" 
                               name="${field.name}" placeholder="${field.placeholder}" 
                               step="any" required>
                        <div class="form-text">${field.help}</div>
                    </div>
                `;
            }
        }).join('');
    }
    
    async makePrediction() {
        const form = document.getElementById('predictionForm');
        const formData = new FormData(form);
        const data = {};
        
        // Convert form data to object
        for (let [key, value] of formData.entries()) {
            if (value.trim()) {
                // Try to convert to number if it's not text field
                if (key !== 'text' && !isNaN(value)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }
        }
        
        const predictBtn = document.getElementById('predictBtn');
        const originalText = predictBtn.innerHTML;
        
        try {
            predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            predictBtn.disabled = true;
            
            let endpoint;
            if (this.currentModel === 'sentiment') {
                endpoint = '/api/predict/nlp/sentiment';
            } else if (this.currentModel === 'keywords') {
                endpoint = '/api/predict/nlp/keywords';
            } else {
                endpoint = `/api/predict/${this.currentModel.replace('_', '-')}`;
            }
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.apiKey
                },
                body: JSON.stringify(data)
            });
            
            let result;
            try {
                result = await response.json();
            } catch (parseError) {
                throw new Error('Server returned invalid JSON response');
            }
            
            if (response.ok) {
                this.displayPredictionResult(result);
                this.showAlert('Prediction completed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            let errorMessage = error.message;
            if (errorMessage.includes('Unexpected token') || errorMessage.includes('JSON')) {
                errorMessage = 'Server returned invalid response. Please check API endpoint.';
            }
            this.showAlert(`Prediction failed: ${errorMessage}`, 'danger');
            this.displayPredictionError(errorMessage);
        } finally {
            predictBtn.innerHTML = originalText;
            predictBtn.disabled = false;
        }
    }
    
    displayPredictionResult(result) {
        const resultsDiv = document.getElementById('predictionResults');
        const prediction = result.prediction;
        
        let html = '';
        
        if (this.currentModel === 'lead_score') {
            html = `
                <div class="prediction-result bg-primary bg-opacity-10 border border-primary">
                    <div class="prediction-score text-primary">${prediction.score}%</div>
                    <div class="h6">Lead Quality: ${prediction.quality}</div>
                    <div class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
                    <small class="text-muted d-block mt-2">Processing time: ${result.processing_time?.toFixed(3)}s</small>
                </div>
            `;
        } else if (this.currentModel === 'churn') {
            const probability = (prediction.churn_probability * 100).toFixed(1);
            html = `
                <div class="prediction-result bg-warning bg-opacity-10 border border-warning">
                    <div class="prediction-score text-warning">${probability}%</div>
                    <div class="h6">Risk Level: ${prediction.risk_level}</div>
                    <div class="prediction-confidence">Will Churn: ${prediction.will_churn ? 'Yes' : 'No'}</div>
                    <small class="text-muted d-block mt-2">${prediction.recommendation}</small>
                </div>
            `;
        } else if (this.currentModel === 'sales_forecast') {
            html = `
                <div class="prediction-result bg-success bg-opacity-10 border border-success">
                    <div class="prediction-score text-success">$${prediction.forecast.toLocaleString()}</div>
                    <div class="h6">Quality: ${prediction.forecast_quality}</div>
                    <div class="prediction-confidence">Range: $${prediction.lower_bound.toLocaleString()} - $${prediction.upper_bound.toLocaleString()}</div>
                    <small class="text-muted d-block mt-2">Period: ${prediction.forecast_period}</small>
                </div>
            `;
        } else if (this.currentModel === 'sentiment') {
            const sentimentClass = prediction.sentiment === 'positive' ? 'success' : 
                                  prediction.sentiment === 'negative' ? 'danger' : 'secondary';
            html = `
                <div class="prediction-result bg-${sentimentClass} bg-opacity-10 border border-${sentimentClass}">
                    <div class="prediction-score text-${sentimentClass}">${prediction.sentiment.toUpperCase()}</div>
                    <div class="h6">Score: ${prediction.score.toFixed(3)}</div>
                    <div class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
                    <small class="text-muted d-block mt-2">Method: ${prediction.method}</small>
                </div>
            `;
        } else if (this.currentModel === 'keywords') {
            const keywords = prediction.keywords.slice(0, 5).map(kw => 
                `<span class="badge bg-info me-1">${kw.keyword} (${kw.score.toFixed(3)})</span>`
            ).join('');
            html = `
                <div class="prediction-result bg-info bg-opacity-10 border border-info">
                    <div class="h6 text-info mb-3">Top Keywords</div>
                    <div class="mb-2">${keywords}</div>
                    <small class="text-muted">Total features: ${prediction.total_features || 'N/A'}</small>
                </div>
            `;
        }
        
        resultsDiv.innerHTML = html;
        
        // Update prediction count
        this.updatePredictionCount();
        
        // Refresh logs if logs tab is active
        const logsTab = document.getElementById('nav-logs');
        if (logsTab && logsTab.classList.contains('active')) {
            setTimeout(() => this.loadPredictionLogs(), 1000);
        }
    }
    
    displayPredictionError(message) {
        const resultsDiv = document.getElementById('predictionResults');
        resultsDiv.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Prediction Error:</strong> ${message}
            </div>
        `;
    }
    
    async uploadFile() {
        const fileInput = document.getElementById('fileInput');
        const modelSelect = document.getElementById('modelSelect');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadProgress = document.getElementById('uploadProgress');
        const progressBar = uploadProgress.querySelector('.progress-bar');
        const progressText = document.getElementById('progressText');
        
        if (!fileInput.files[0] || !modelSelect.value) {
            this.showAlert('Please select a file and model type', 'warning');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model_type', modelSelect.value);
        
        try {
            uploadBtn.disabled = true;
            uploadProgress.style.display = 'block';
            progressBar.style.width = '0%';
            progressText.textContent = 'Uploading file...';
            
            // Simulate progress during upload
            const progressInterval = setInterval(() => {
                const currentWidth = parseFloat(progressBar.style.width) || 0;
                if (currentWidth < 90) {
                    progressBar.style.width = (currentWidth + 10) + '%';
                }
            }, 200);
            
            const response = await fetch('/api/upload/', {
                method: 'POST',
                headers: {
                    'X-API-Key': this.apiKey
                },
                body: formData
            });
            
            clearInterval(progressInterval);
            const result = await response.json();
            
            if (response.ok) {
                progressBar.style.width = '100%';
                progressText.textContent = 'Upload complete!';
                
                this.showAlert(`File uploaded successfully! ${result.total_rows} rows detected.`, 'success');
                
                // Start processing
                setTimeout(() => {
                    this.processUploadedFile(result.upload_id);
                }, 1000);
                
                // Reset form
                fileInput.value = '';
                modelSelect.value = '';
                
            } else {
                throw new Error(result.error || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert(`Upload failed: ${error.message}`, 'danger');
            progressText.textContent = 'Upload failed';
        } finally {
            uploadBtn.disabled = false;
            setTimeout(() => {
                uploadProgress.style.display = 'none';
            }, 2000);
        }
    }
    
    async processUploadedFile(uploadId) {
        try {
            const response = await fetch(`/api/upload/${uploadId}/process`, {
                method: 'POST',
                headers: {
                    'X-API-Key': this.apiKey
                }
            });
            
            let result;
            try {
                result = await response.json();
            } catch (parseError) {
                throw new Error('Server returned invalid JSON response');
            }
            
            if (response.ok) {
                this.showAlert('File processing started! Check upload history for progress.', 'info');
                this.loadUploadHistory();
                this.updateUploadCount();
            } else {
                throw new Error(result.error || 'Processing failed');
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            let errorMessage = error.message;
            if (errorMessage.includes('Unexpected token')) {
                errorMessage = 'Server returned invalid response. Please check API endpoint.';
            }
            this.showAlert(`Processing failed: ${errorMessage}`, 'danger');
        }
    }
    
    async loadUploadHistory() {
        try {
            const response = await fetch('/api/upload/', {
                headers: {
                    'X-API-Key': this.apiKey
                }
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayUploadHistory(result.uploads);
            } else {
                throw new Error(result.error || 'Failed to load upload history');
            }
            
        } catch (error) {
            console.error('Upload history error:', error);
            const historyDiv = document.getElementById('uploadHistory');
            if (historyDiv) {
                historyDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Failed to load upload history
                    </div>
                `;
            }
        }
    }
    
    displayUploadHistory(uploads) {
        const historyDiv = document.getElementById('uploadHistory');
        
        if (!uploads || uploads.length === 0) {
            historyDiv.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-inbox fa-3x mb-3 opacity-50"></i>
                    <p>No uploads yet</p>
                </div>
            `;
            return;
        }
        
        const html = uploads.slice(0, 5).map(upload => {
            const statusClass = upload.status === 'completed' ? 'success' : 
                               upload.status === 'error' ? 'danger' : 
                               upload.status === 'processing' ? 'warning' : 'secondary';
            
            const progress = upload.total_rows > 0 ? (upload.processed_rows / upload.total_rows * 100).toFixed(1) : 0;
            
            return `
                <div class="card mb-2">
                    <div class="card-body py-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="mb-1">${upload.original_filename}</h6>
                                <small class="text-muted">
                                    ${upload.model_type} • ${upload.total_rows} rows
                                </small>
                            </div>
                            <div class="text-end">
                                <span class="badge bg-${statusClass}">${upload.status}</span>
                                ${upload.status === 'processing' ? `<div class="small text-muted">${progress}%</div>` : ''}
                                ${upload.has_results ? `<a href="/api/upload/${upload.id}/download" class="btn btn-sm btn-outline-primary mt-1">Download</a>` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        historyDiv.innerHTML = html;
    }
    
    async loadPredictionLogs() {
        try {
            const response = await fetch('/api/predict/logs', {
                headers: {
                    'X-API-Key': this.apiKey
                }
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayPredictionLogs(result.logs);
            } else {
                throw new Error(result.error || 'Failed to load logs');
            }
            
        } catch (error) {
            console.error('Logs error:', error);
            const tbody = document.getElementById('logsTableBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center text-danger">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Failed to load prediction logs
                        </td>
                    </tr>
                `;
            }
        }
    }
    
    displayPredictionLogs(logs) {
        const tbody = document.getElementById('logsTableBody');
        
        if (!logs || logs.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-muted">
                        <i class="fas fa-inbox me-2"></i>
                        No prediction logs found
                    </td>
                </tr>
            `;
            return;
        }
        
        const html = logs.map(log => {
            const statusClass = log.status === 'success' ? 'success' : 'danger';
            const confidence = log.confidence ? (log.confidence * 100).toFixed(1) + '%' : 'N/A';
            const processingTime = log.processing_time ? log.processing_time.toFixed(3) + 's' : 'N/A';
            const createdAt = new Date(log.created_at).toLocaleString();
            
            return `
                <tr>
                    <td>${createdAt}</td>
                    <td>
                        <span class="badge bg-secondary">${log.model_type}</span>
                    </td>
                    <td>
                        <span class="badge bg-${statusClass}">${log.status}</span>
                    </td>
                    <td>${confidence}</td>
                    <td>${processingTime}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-secondary" onclick="dashboard.showLogDetails(${log.id})">
                            <i class="fas fa-eye"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
        
        tbody.innerHTML = html;
    }
    
    async loadCrmData(platform) {
        const dataDiv = document.getElementById(`${platform}Data`);
        const loadBtn = document.getElementById(`load${platform.charAt(0).toUpperCase() + platform.slice(1)}Leads`);
        
        try {
            loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
            loadBtn.disabled = true;
            
            const response = await fetch(`/api/connectors/${platform}/leads`, {
                headers: {
                    'X-API-Key': this.apiKey
                }
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayCrmData(dataDiv, result.leads, platform);
                this.showAlert(`Loaded ${result.count} leads from ${platform.charAt(0).toUpperCase() + platform.slice(1)}`, 'success');
            } else {
                throw new Error(result.error || `Failed to load ${platform} data`);
            }
            
        } catch (error) {
            console.error(`${platform} error:`, error);
            dataDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${error.message}
                </div>
            `;
        } finally {
            loadBtn.innerHTML = `<i class="fas fa-download me-2"></i>Load Leads`;
            loadBtn.disabled = false;
        }
    }
    
    displayCrmData(container, leads, platform) {
        if (!leads || leads.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-inbox fa-2x mb-2 opacity-50"></i>
                    <p>No leads found</p>
                </div>
            `;
            return;
        }
        
        const html = `
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Company</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${leads.slice(0, 10).map(lead => `
                            <tr>
                                <td>${lead.name || 'N/A'}</td>
                                <td>${lead.email || 'N/A'}</td>
                                <td>${lead.company || 'N/A'}</td>
                                <td>
                                    <span class="badge bg-${lead.score > 70 ? 'success' : lead.score > 40 ? 'warning' : 'secondary'}">
                                        ${lead.score}
                                    </span>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ${leads.length > 10 ? `<small class="text-muted">Showing 10 of ${leads.length} leads</small>` : ''}
        `;
        
        container.innerHTML = html;
    }
    
    async checkApiHealth() {
        try {
            const response = await fetch('/health');
            const result = await response.json();
            
            if (response.ok) {
                document.getElementById('apiStatus').textContent = 'Healthy';
            } else {
                document.getElementById('apiStatus').textContent = 'Error';
            }
        } catch (error) {
            console.error('Health check error:', error);
            document.getElementById('apiStatus').textContent = 'Offline';
        }
    }
    
    updatePredictionCount() {
        const countElement = document.getElementById('predictionCount');
        if (countElement) {
            const currentCount = parseInt(countElement.textContent) || 0;
            countElement.textContent = currentCount + 1;
        }
    }
    
    updateUploadCount() {
        const countElement = document.getElementById('uploadCount');
        if (countElement) {
            const currentCount = parseInt(countElement.textContent) || 0;
            countElement.textContent = currentCount + 1;
        }
    }
    
    copyApiKey() {
        const apiKeyInput = document.getElementById('defaultApiKey');
        apiKeyInput.select();
        apiKeyInput.setSelectionRange(0, 99999);
        
        try {
            document.execCommand('copy');
            this.showAlert('API key copied to clipboard!', 'success');
        } catch (error) {
            this.showAlert('Failed to copy API key', 'danger');
        }
    }
    
    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = 'alert-' + Date.now();
        
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
                <i class="fas fa-${type === 'success' ? 'check-circle' : 
                                    type === 'danger' ? 'exclamation-triangle' : 
                                    type === 'warning' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        alertContainer.innerHTML = alertHtml;
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
    
    initializeCharts() {
        // Check if Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded, skipping chart initialization');
            return;
        }
        
        // Performance Chart
        const performanceCtx = document.getElementById('performanceChart');
        if (performanceCtx) {
            this.charts.performance = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [{
                        label: 'Predictions',
                        data: [12, 19, 3, 5, 2, 3],
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        // Usage Chart
        const usageCtx = document.getElementById('usageChart');
        if (usageCtx) {
            this.charts.usage = new Chart(usageCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Lead Scoring', 'Churn Prediction', 'Sales Forecast', 'NLP'],
                    datasets: [{
                        data: [30, 25, 25, 20],
                        backgroundColor: [
                            '#0d6efd',
                            '#6f42c1',
                            '#20c997',
                            '#fd7e14'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }
    
    showLogDetails(logId) {
        // This would show detailed log information in a modal
        this.showAlert('Log details functionality coming soon!', 'info');
    }
}

// Global dashboard instance
let dashboard;

function initializeDashboard() {
    // Wait for Chart.js to be fully loaded
    const initDashboard = () => {
        try {
            dashboard = new DashboardManager();
            
            // Initialize first model
            dashboard.updateInputForm();
        } catch (error) {
            console.error('Dashboard initialization error:', error);
        }
    };
    
    // Check if Chart.js is loaded, if not wait a bit
    if (typeof Chart !== 'undefined') {
        initDashboard();
    } else {
        setTimeout(() => {
            initDashboard();
        }, 500);
    }
}

// Export for global access
window.dashboard = dashboard;

// Load credit information
function loadCreditInfo() {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        document.getElementById('credits-remaining').textContent = 'Login Required';
        document.getElementById('plan-name').textContent = 'N/A';
        return;
    }
    
    fetch('/api/auth/credits', {
        headers: {
            'Authorization': 'Bearer ' + token
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.credits_remaining !== undefined) {
            document.getElementById('credits-remaining').textContent = data.credits_remaining;
            document.getElementById('plan-name').textContent = data.plan_name;
            
            const progressPercent = (data.credits_remaining / data.plan_limit) * 100;
            const progressBar = document.getElementById('credit-progress');
            progressBar.style.width = progressPercent + '%';
            
            if (progressPercent < 20) {
                progressBar.className = 'progress-bar bg-danger';
            } else if (progressPercent < 50) {
                progressBar.className = 'progress-bar bg-warning';
            }
        }
    })
    .catch(error => {
        console.log('Credit info not available:', error);
        document.getElementById('credits-remaining').textContent = '100';
        document.getElementById('plan-name').textContent = 'Free';
    });
}

// Call loadCreditInfo on page load if elements exist
if (document.getElementById('credits-remaining')) {
    loadCreditInfo();
}
