Doctor Survey Targeting System 🏥
A machine learning-powered system that predicts which doctors are most likely to attend surveys/seminars at specific hours based on their historical login patterns, usage behavior, and engagement metrics. This project enables targeted email campaigns and optimized scheduling for healthcare platforms.

📋 Table of Contents

Overview
Problem Statement
Dataset
Key Features
Technology Stack
System Architecture
Installation
Usage
Analysis Workflow
Model Performance
Key Findings
Future Enhancements


🎯 Overview
The Doctor Survey Targeting System is an intelligent healthcare analytics platform that analyzes doctor engagement patterns to predict optimal times for conducting surveys and seminars. By leveraging machine learning algorithms and historical activity data, the system identifies which doctors are most likely to be active at specific hours, enabling:

Targeted Email Campaigns: Send invitations at optimal times for maximum response
Resource Optimization: Schedule events when target doctors are most available
Improved Engagement: Increase survey participation rates through data-driven targeting
Time-Zone Awareness: Account for regional differences in doctor availability

🎓 Problem Statement
Healthcare platforms conducting doctor surveys face significant challenges:

Low Response Rates: Generic survey invitations result in poor participation
Timing Mismatch: Surveys sent when doctors are offline or busy
Resource Wastage: Inefficient use of communication channels
Limited Insights: No data-driven approach to understand doctor availability patterns

Solution: Predict the top 20 doctors most likely to be active at any given hour, enabling targeted outreach with up to 99.3% AUC accuracy.
📊 Dataset
Source
Dummy dataset of 1,000 US doctors (NPIs) collected from a healthcare platform on March 8, 2025.
Structure

Rows: 1,000 (one session per doctor)
Columns: 8 core features + 15 engineered features

Core Features
FeatureTypeDescriptionNPIIntegerUnique National Provider IdentifierStateCategoricalUS State (e.g., NY, CA, TX)RegionCategoricalGeographic region (Northeast, Midwest, South, West)SpecialityCategoricalMedical specialty (Oncology, Cardiology, etc.)Login TimeDatetimeSession start timestampLogout TimeDatetimeSession end timestampUsage Time (mins)IntegerSession duration (5-120 minutes)Count of Survey AttemptsIntegerPast week survey attempts (0-10)
<img width="1829" height="510" alt="image" src="https://github.com/user-attachments/assets/0fa06193-7afc-4f10-8833-86d6db2ceaab" />

Sample Data Statistics

Mean Session Duration: 64.6 minutes
Peak Login Hour: 13:00 (1 PM) - 81 doctors
Average Survey Attempts: 5 per week
Most Common Specialty: Oncology (~200 doctors)
Time Range: 06:00 - 22:46 (across all sessions)

✨ Key Features
1. Intelligent Time Prediction

Predicts top 20 doctors likely to attend at any specified hour (0-23)
Considers cyclic nature of time using sine/cosine transformations
Accounts for individual doctor patterns and specialty-specific behaviors

2. Advanced Feature Engineering

Cyclic Features: Login/Logout hour encoded as sin/cos for circular time
Aggregate Metrics: Average usage time per specialty, login hour variability
Regional Insights: Average survey attempts by region
Engagement Rate: Historical attendance rate as proxy for participation

3. Multi-Model Approach

XGBoost Classifier (Primary - Python): AUC = 0.993, F1 = 0.845
Random Forest Classifier (R Implementation): Precision/Recall analysis
K-Means Clustering: Segments doctors into 4 engagement groups

4. Interactive Web Interface

Streamlit Application: User-friendly hour selection and prediction display
Real-Time Predictions: Instant results with probability scores
Export Functionality: Download top doctors list as CSV

5. Comprehensive Visualizations

Login/logout activity heatmaps by hour
Feature correlation matrices
PCA cluster analysis in 2D space
Regional and specialty distribution plots

<img width="795" height="564" alt="image" src="https://github.com/user-attachments/assets/3dd5d79a-f0a4-4748-aa61-6e88d075db51" />

<img width="810" height="683" alt="image" src="https://github.com/user-attachments/assets/bf4180e6-fa1e-4b31-bbaf-0ca68f491d85" />


🛠️ Technology Stack
Python Stack

Core: Python 3.8+, Pandas, NumPy
ML Libraries: XGBoost, scikit-learn
Visualization: Matplotlib, Seaborn
Web Framework: Streamlit (UI), Flask (API)
Environment: Google Colab, Jupyter Notebook

R Stack

Core: R 4.0+
Libraries:

randomForest - Model training
ggplot2 - Visualizations
lubridate - Date/time handling
tidyverse - Data manipulation
corrplot - Correlation analysis
readxl - Excel import



Development Tools

IDE: VS Code, RStudio
Version Control: Git, GitHub
Documentation: LaTeX (for academic report)

🏗️ System Architecture
Data Flow Pipeline
Raw Data → Cleaning → Feature Engineering → Model Training → Prediction → Web UI
   ↓           ↓              ↓                    ↓             ↓          ↓
1000 rows   DateTime    15 new features      XGBoost/RF    Top 20 NPIs  Streamlit
           conversion   (cyclic, agg.)       GridSearch    w/ probs     + Export
System Components

Data Collection Module: Loads CSV/Excel files, validates schema
Preprocessing Pipeline: Handles missing values, type conversion, encoding
Feature Engineering: Creates cyclic, aggregate, and temporal features
Model Training:

Expands dataset to 24,000 rows (1000 doctors × 24 hours)
Trains with StratifiedKFold cross-validation
Hyperparameter tuning via GridSearchCV


Prediction Engine: Ranks doctors by probability for target hour
Web Interface: Streamlit app for user interaction and CSV export

<img width="802" height="518" alt="image" src="https://github.com/user-attachments/assets/162685be-1387-4442-ae43-74c7e3f1bd17" />

📥 Installation
Prerequisites
bashPython 3.8 or higher
R 4.0 or higher (optional, for R implementation)
pip or conda
Setup Instructions
Python Environment
bash# Clone the repository
git clone https://github.com/poorvapathak/Doctor_Survey.git
cd Doctor_Survey

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
```
pip install -r requirements.txt
```
# If requirements.txt is missing, install manually:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
R Environment (Optional)
R# Install required packages in RStudio
install.packages(c("tidyverse", "randomForest", "corrplot", 
                   "lubridate", "janitor", "readxl", "ggplot2"))
```

### File Structure
```
Doctor_Survey/
├── Doctor_Survey_Target (2).ipynb         # Python EDA & ML notebook
├── dummy_npi_data.xlsx - Dataset.csv     # Raw dataset
├── R_Report_Doctor_Survey_Prediction.pdf # Academic report
├── README.md                              # This file
├── top_doctors.csv                        # Sample prediction output
├── requirements.txt                       # Python dependencies (if available)
└── streamlit_app.py                       # Web UI (if separated from notebook)
```
🚀 Usage
Running the Jupyter Notebook
bash# Start Jupyter
jupyter notebook

# Open Doctor_Survey_Target (2).ipynb
# Run all cells sequentially (Cell → Run All)
```
Using the Prediction Function
pythonimport pandas as pd
import numpy as np
from xgboost import XGBClassifier
```

# Load trained model (from notebook)
# best_model = ... (trained XGBoost pipeline)

# Example: Predict for 10 AM
```
hour_input = 10
top_doctors = predict_likely_doctors_at_hour(hour_input, df, best_model)

print(f"Top 20 doctors likely to attend at {hour_input}:00:")
print(top_doctors[['NPI', 'State', 'Region', 'Speciality', 'Probability']])

# Export results
top_doctors.to_csv('predictions_10am.csv', index=False)
Running the Streamlit App
bash# If app is in separate file
streamlit run streamlit_app.py
```

# Or from notebook (if integrated)
# Navigate to the Streamlit section and run the app cells
<img width="823" height="590" alt="image" src="https://github.com/user-attachments/assets/08575e07-92a6-446c-8b22-adc916886136" />

Web Interface Usage

Input Hour: Select hour (0-23) using slider or text input
View Predictions: See ranked list of top 20 doctors with:

NPI (identifier)
State and Region
Medical Specialty
Probability Score (0.83-0.88 for top ranks)


Download Results: Export as CSV for email campaigns

🔬 Analysis Workflow
Step 1: Data Exploration

Load 1,000 doctor sessions from March 8, 2025
Check for missing values (none found) and duplicates (none)
Analyze distributions: regions balanced (~250 each), Oncology dominant

<img width="1036" height="696" alt="image" src="https://github.com/user-attachments/assets/e6ed99fe-df01-4013-8524-5787dfa33aa4" />

<img width="1273" height="808" alt="image" src="https://github.com/user-attachments/assets/c42937ac-0e22-437e-afce-dccae143d969" />

Step 2: Temporal Analysis

Convert login/logout times to datetime objects
Extract hour features (6:00 - 22:46 range)
Identify peak activity: Logins at 13:00 (81 doctors), Logouts at 10:00 (68 doctors)

Key Insight: Midday (12-15:00) is optimal for survey campaigns.
Step 3: Feature Engineering
Created 15 new features:

Cyclic Time Features: Login_Sin_Hour, Login_Cos_Hour (captures 24-hour cycle)
Aggregations:

Avg_Usage_Time by Specialty (62.7-69.1 mins)
Std_Login_Hour by Specialty (~4.3 hours)
Avg_Attempts_By_Region (4.76-5.20)


Engagement Metrics: Historical_Attendance_Rate (~0.077)

Step 4: Dataset Expansion

Expanded from 1,000 rows to 24,000 rows (1000 doctors × 24 hours)
Created binary target: Active=1 if session overlaps target hour (±30 min window)
Class distribution: Active=8.5%, Inactive=91.5% (imbalanced, handled via stratification)

Step 5: Correlation Analysis
<img width="804" height="698" alt="image" src="https://github.com/user-attachments/assets/a5cbb3b4-08fa-4acc-90ef-853e3008bd3b" />

Findings:

Weak correlations overall (max ~0.3)
Usage time slightly correlates with survey attempts
Login/logout hours show minimal direct correlation

Step 6: Model Training
XGBoost (Primary Model)

Data Split: 75% train, 25% test (stratified)
Hyperparameter Tuning: GridSearchCV with StratifiedKFold (5 splits)
Best Parameters:

learning_rate: 0.1
max_depth: 5
n_estimators: 100
subsample: 0.8



Random Forest (Alternative - R)

Implemented in R for comparison
Feature importance analysis
Cross-validation for robustness

Step 7: Clustering Analysis

Algorithm: K-Means with k=4 (determined via Elbow method)
Features: PCA components, survey rate, usage time
Clusters Identified:

Cluster 0: High attempts + High usage
Cluster 1: Low attempts + Low usage
Cluster 2: Moderate engagement
Cluster 3: High attempts + Low usage (efficient responders)

<img width="453" height="264" alt="image" src="https://github.com/user-attachments/assets/061a5847-fe7a-475e-9a35-c1d956cfa01e" />

<img width="458" height="368" alt="image" src="https://github.com/user-attachments/assets/170225fe-f601-4f45-8d88-0a376c5bb19a" />

📈 Model Performance
XGBoost Classifier Results
MetricScoreAUC-ROC0.9932F1 Score0.8448Precision0.8424Precision@201.0000
<img width="484" height="378" alt="image" src="https://github.com/user-attachments/assets/b84d8374-03c3-4e1f-a291-1dcb0265a9df" />

Interpretation

AUC = 0.993: Excellent discrimination between active/inactive doctors
Precision@20 = 1.0: All top 20 predictions are true positives (reliable for campaigns!)
F1 = 0.845: Strong balance between precision and recall

Random Forest (R) Results

Achieved comparable precision/recall
Validated XGBoost findings
Provided feature importance insights (usage time, cyclic features top-ranked)

Cross-Validation

5-fold stratified CV ensures generalizability
Consistent performance across folds (~0.99 AUC)
No overfitting detected

🔍 Key Findings
1. Optimal Targeting Windows

Peak Hours: 12:00-15:00 (midday) for maximum doctor availability
Secondary Peak: 20:00 (evening) - 75 logins
Low Activity: 16:00 (53 logins) and early morning (6:00-8:00)

2. Specialty Insights

Oncology: Most common (~200 doctors), high engagement
Pediatrics & Orthopedics: Frequently appear in top predictions
General Practice: Moderate engagement, longer sessions

3. Regional Patterns

Balanced Distribution: West/Midwest ~250 doctors each
Engagement Consistency: Regional differences minimal (4.76-5.20 avg attempts)
Top Prediction Regions: South and West appear frequently in top 20

4. Usage Behavior

Average Session: 64.6 minutes (range: 5-120 mins)
Trend Lag: Sessions end ~1 hour after login on average
Survey Attempts: Mean 5 per week; 0 attempts indicate inactive doctors

5. Prediction Accuracy

Top 20 Doctors: Probabilities range 0.83-0.88 (high confidence)
Example (10 AM):

NPI 1000000131 (CA/South/Pediatrics) - 88.1% probability
NPI 1000000323 (MI/South/Pediatrics) - 87.1% probability

<img width="470" height="304" alt="image" src="https://github.com/user-attachments/assets/f6af4d83-95f7-45b9-8b0a-7b9200db60ea" />

6. Actionable Recommendations

Schedule surveys at 13:00 (1 PM) for maximum reach
Target Oncology and Pediatrics specialists first
Send invitations 8-11 AM to catch doctors as they log in
Leverage 64-min session window for follow-ups

🔮 Future Enhancements
Data Enrichment

 Multi-Day Historical Data: Capture weekly/monthly patterns
 External Data Sources: Demographics, patient population trends
 Real-Time Integration: Live availability status, appointment schedules
 Weather/Event Data: Correlate with external factors affecting attendance

Model Improvements

 Deep Learning: LSTM for sequential time-series prediction
 Gradient Boosting Variants: LightGBM, CatBoost for comparison
 Ensemble Methods: Combine XGBoost + Random Forest predictions
 Explainability: SHAP values for interpreting individual predictions

Application Expansion

 Broader Applicability: Adapt for customer engagement (retail, online platforms)
 A/B Testing: Validate predictions with real campaign data
 Mobile App: iOS/Android interface for on-the-go predictions
 API Development: REST API for integration with healthcare platforms

Ethical Considerations

 Bias Mitigation: Ensure fairness across specialties and regions
 Privacy Enhancement: Implement differential privacy techniques
 Consent Management: Transparent data usage policies
 Accountability: Audit trails for prediction decisions

UI/UX Enhancements

 Interactive Dashboards: Plotly/Dash for dynamic exploration
 Batch Predictions: Upload doctor list, get probabilities for all
 Email Integration: Direct campaign launch from interface
 Reporting: Automated PDF reports with insights


🙏 Acknowledgments

Kaggle Community: For inspiration on time-series prediction workflows
Streamlit: For enabling rapid prototyping of ML interfaces
XGBoost Team: For the powerful gradient boosting library


⭐ If you find this project useful for healthcare analytics or time-based prediction tasks, please star the repository!
📚 References

Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.
Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
McKinney, W. (2011). pandas: A Foundational Python Library for Data Analysis. PyHPC 2011.


Project Status: ✅ Complete | Last Updated: March 2025 | Version: 1.0
