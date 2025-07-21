# Data Requirements & Handling Guide

## Overview

This AI Prediction Platform is designed to handle both clean and messy data. While we can process uncleaned data with missing values and inconsistencies, **we strongly recommend cleaning your data first** for optimal prediction accuracy.

## Data Handling Philosophy

✅ **We Accept**: Uncleaned data, missing values, inconsistent formats  
⚠️ **We Recommend**: Clean, well-formatted data for best results  
🎯 **We Deliver**: Robust predictions regardless of data quality  

---

## Supported File Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **Maximum file size**: 16MB
- **Maximum rows**: 100,000 rows per file

---

## Data Requirements by Model Type

### 1. Lead Scoring Model

#### Required Columns (any of these column names):
```
company_size, budget, industry_score, engagement_score, 
demographic_score, behavioral_score, lead_source, 
company_revenue, employee_count, website_traffic
```

#### Data Handling:
- **Missing values**: Automatically filled with median/mode values
- **Text fields**: Converted to numerical scores where possible
- **Inconsistent formats**: Normalized automatically
- **Duplicate records**: Kept (may indicate lead strength)

#### Example Data Formats We Accept:

**Clean Data (Preferred):**
```csv
company_size,budget,industry_score,engagement_score,demographic_score
100,50000,4,3,4
250,75000,5,4,3
50,25000,3,2,4
```

**Messy Data (We Handle This):**
```csv
company_size,budget,industry_score,engagement_score,demographic_score
"100 employees",50k,4,3,4
"250-300",$75000,5,,3
,25000,3,2,4
"Small company",,3,2,
```

### 2. Churn Prediction Model

#### Required Columns:
```
customer_tenure, monthly_charges, total_charges, contract_type,
payment_method, customer_age, support_tickets, usage_frequency,
satisfaction_score, last_interaction_days
```

#### Data Handling:
- **Missing tenure**: Estimated from other customer data
- **Invalid charges**: Set to account average
- **Missing satisfaction**: Neutral score assigned
- **Text categories**: Auto-encoded to numerical values

#### Example Data Formats:

**Clean Data:**
```csv
customer_tenure,monthly_charges,total_charges,contract_type
24,89.99,2159.76,Two year
12,65.50,786.00,One year
6,45.25,271.50,Monthly
```

**Messy Data We Handle:**
```csv
customer_tenure,monthly_charges,total_charges,contract_type
"2 years",89.99,2159.76,Two year
12,"$65.50","$786",One year
,"45.25",,"Monthly"
"6 months",,"271.50",
```

### 3. Sales Forecast Model

#### Required Columns:
```
date, sales_amount, product_category, region, marketing_spend,
lead_volume, conversion_rate, seasonal_factor, economic_indicator,
competitor_activity, weather_impact
```

#### Data Handling:
- **Missing dates**: Interpolated from sequence
- **Invalid sales amounts**: Replaced with trend estimates
- **Missing categorical data**: Assigned to "Other" category
- **Seasonal adjustments**: Applied automatically

#### Example Data Formats:

**Clean Data:**
```csv
date,sales_amount,product_category,region,marketing_spend
2023-01-01,15000,Software,North,5000
2023-01-02,16500,Software,South,5200
2023-01-03,14200,Hardware,West,4800
```

**Messy Data We Handle:**
```csv
date,sales_amount,product_category,region,marketing_spend
"Jan 1, 2023","$15,000",Software,North,"5000"
2023-01-02,16500,,South,
"1/3/23",14200,Hardware,West,4800
,,"Software",,5000
```

### 4. Sentiment Analysis Model

#### Required Columns:
```
text, customer_id, date, product, rating, review_source,
verified_purchase, helpful_votes, review_length
```

#### Data Handling:
- **Empty text**: Skipped with warning
- **Mixed languages**: Auto-detected and processed
- **HTML/Special chars**: Cleaned automatically
- **Duplicate reviews**: Flagged but processed

#### Example Data Formats:

**Clean Data:**
```csv
text,customer_id,rating,product
"Great product, highly recommend",CUST001,5,Widget Pro
"Not what I expected, poor quality",CUST002,2,Widget Basic
```

**Messy Data We Handle:**
```csv
text,customer_id,rating,product
"Great product!!! ❤️ highly recommend 😊",CUST001,5,Widget Pro
"Not what I expected... poor quality :(",CUST002,"2 stars",
"<p>Amazing service</p>",,"5",Widget Pro
,CUST003,4,
```

### 5. Keyword Extraction Model

#### Required Columns:
```
text, document_id, category, source, date_created,
language, word_count, importance_score
```

#### Data Handling:
- **Short text**: Processed with limited keywords
- **Multi-language**: Separated by language
- **Special characters**: Preserved for context
- **HTML content**: Tags removed, text extracted

---

## Data Quality Recommendations

### ⭐ Best Practices for Optimal Results

1. **Clean Missing Values**
   ```
   ❌ NULL, "", "N/A", "Unknown", "TBD"
   ✅ Remove rows or fill with appropriate values
   ```

2. **Standardize Formats**
   ```
   ❌ Mixed date formats: "1/1/23", "Jan 1, 2023", "2023-01-01"
   ✅ Single format: "2023-01-01"
   ```

3. **Normalize Text Data**
   ```
   ❌ "Yes", "YES", "Y", "True", "1"
   ✅ All "Yes" or all "1"
   ```

4. **Remove Duplicates**
   ```
   ❌ Same customer/transaction multiple times
   ✅ One record per unique entity
   ```

5. **Validate Ranges**
   ```
   ❌ Age: -5, 250, "old"
   ✅ Age: 25, 45, 67
   ```

### 🛠️ Pre-Processing Tools We Recommend

- **Excel**: Data validation, remove duplicates, find/replace
- **Python**: pandas.dropna(), fillna(), to_datetime()
- **R**: na.omit(), complete.cases(), str_trim()
- **SQL**: COALESCE, NULLIF, TRIM functions

---

## What Happens During Processing

### Step 1: Data Ingestion
- File format validation
- Encoding detection (UTF-8, ASCII, etc.)
- Structure analysis (columns, rows, data types)

### Step 2: Data Assessment
- Missing value analysis
- Outlier detection
- Data type inference
- Quality score calculation

### Step 3: Auto-Cleaning Pipeline
- **Missing Numerics**: Filled with median values
- **Missing Categories**: Assigned to "Other" or most frequent
- **Invalid Dates**: Converted or removed
- **Text Normalization**: Case standardization, trim whitespace
- **Outlier Handling**: Capped at percentile boundaries

### Step 4: Feature Engineering
- **Categorical Encoding**: One-hot or label encoding
- **Date Features**: Extract day, month, year, weekday
- **Text Features**: TF-IDF, sentiment scores, length metrics
- **Numerical Scaling**: Standardization or normalization

### Step 5: Model-Specific Processing
- **Lead Scoring**: Score binning, interaction features
- **Churn**: Tenure grouping, usage patterns
- **Sales Forecast**: Trend decomposition, seasonality
- **Sentiment**: Token extraction, emotion detection
- **Keywords**: TF-IDF scoring, phrase detection

---

## Error Handling & Notifications

### Warnings You'll Receive:
- `Missing data filled with default values`
- `Invalid date formats converted`
- `Text contains special characters (cleaned)`
- `Duplicate records detected`
- `Columns renamed for compatibility`

### Errors That Stop Processing:
- `File format not supported`
- `No valid data rows found`
- `Required columns missing`
- `File size exceeds limit`
- `Too many missing values (>90%)`

---

## Data Upload Guidelines

### Before Upload (Recommended):

1. **Review Your Data**
   - Check for obvious errors
   - Remove test/dummy records
   - Validate important dates/amounts

2. **Basic Cleaning**
   - Remove empty rows at the end
   - Check column headers are descriptive
   - Ensure numeric fields don't have text

3. **Save Backup**
   - Keep original file
   - Document any changes made

### During Upload:

1. **Select Correct Model Type**
   - Match your data purpose to the right model
   - Check column requirements above

2. **Monitor Progress**
   - Watch for validation messages
   - Note any warnings or suggestions

3. **Review Results**
   - Check the data quality report
   - Verify sample predictions make sense

### After Processing:

1. **Download Results**
   - Get processed data file
   - Review prediction confidence scores
   - Check any flagged records

2. **Iterate if Needed**
   - Clean flagged issues
   - Re-upload for better accuracy
   - Compare prediction improvements

---

## API Integration for Developers

### Headers Required:
```http
X-API-Key: your-api-key-here
Content-Type: application/json  # For individual predictions
Content-Type: multipart/form-data  # For file uploads
```

### Individual Prediction Example:
```javascript
// Even with messy data, the API handles it
const messyData = {
    "company_size": "50-100 employees",  // Will be normalized
    "budget": "$25,000",                 // Will be cleaned  
    "industry_score": "",                // Will be filled
    "engagement_score": "high"           // Will be converted
};

fetch('/api/predict/lead-score', {
    method: 'POST',
    headers: {
        'X-API-Key': 'your-key',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(messyData)
});
```

### File Upload Example:
```javascript
const formData = new FormData();
formData.append('file', messyCSVFile);  // Can contain missing values
formData.append('model_type', 'lead_score');

fetch('/api/upload/', {
    method: 'POST',
    headers: {
        'X-API-Key': 'your-key'
    },
    body: formData
});
```

---

## Troubleshooting Common Issues

### "No valid data found"
**Cause**: All rows are empty or all required columns missing  
**Solution**: Ensure at least one column matches model requirements

### "Too many missing values"
**Cause**: >90% of critical fields are empty  
**Solution**: Fill more values or use a different model type

### "File format error"
**Cause**: Corrupted file or unsupported format  
**Solution**: Re-save as CSV or check file integrity

### "Processing timeout"
**Cause**: File too large or complex  
**Solution**: Split into smaller files or remove unnecessary columns

### "Low prediction confidence"
**Cause**: Data quality issues affecting model accuracy  
**Solution**: Clean more data points and re-upload

---

## Support & Best Practices

### Getting Help:
- Check processing logs in the dashboard
- Review data quality scores
- Use the API testing interface
- Contact support with specific error messages

### Continuous Improvement:
- Monitor prediction accuracy over time
- Update data regularly
- Clean incrementally for better results
- Track which data cleaning steps help most

### Performance Tips:
- Smaller files process faster
- Fewer missing values = better predictions
- Standard formats reduce processing time
- Regular updates improve model accuracy

---

**Remember**: While we handle messy data gracefully, clean data always produces better predictions. The extra time spent on data preparation will result in more accurate and reliable AI insights.