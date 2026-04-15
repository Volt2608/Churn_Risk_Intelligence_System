### Customer Churn Prediction System

### Overview
This project builds an end-to-end machine learning system to predict customer churn using real-world banking data. The goal is to identify customers at risk of leaving and enable proactive retention strategies.

---

### Problem Statement
Customer churn directly impacts business revenue. This project aims to predict churn using demographic, financial, and behavioral data, with a focus on minimizing missed churn cases.

---

### Project Structure
    churn_risk_system/

    │
    ├── data/
    ├── notebooks/
    │ ├── 1_eda.ipynb
    │ ├── 2_data_preprocessing.ipynb
    │ ├── 3_modeling.ipynb
    │
    ├── models/
    │ ├── train.py
    │ ├── evaluate.py
    │ ├── predict.py
    │ ├── churn_model.pkl
    │ ├── scaler.pkl
    │ ├── feature_names.pkl
    │
    ├── app/
    │ ├── main.py
    │
    ├── requirements.txt
    ├── README.md

---

### Approach

#### 1. Exploratory Data Analysis
- Identified churn patterns across geography, age, and customer activity  
- Observed moderate class imbalance (~80/20)

### 2. Data Preprocessing
- One-hot encoding for categorical variables  
- Feature scaling using StandardScaler  
- Stratified train-test split  

### 3. Modeling
- Logistic Regression (baseline)  
- Random Forest (advanced model)

### 4. Model Optimization
- Applied threshold tuning to optimize recall vs precision  
- Used F2 score to prioritize churn detection  

### 5. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  
- F2 Score  

---

### Key Results

| Model | Recall | F2 Score | Insight |
|------|--------|---------|--------|
| Logistic Regression | 0.83 | 0.63 | Best for detecting churn |
| Random Forest | 0.67 | 0.65 | Balanced performance |

---

### Feature Importance
Key drivers of churn:
- Age  
- Balance  
- Estimated Salary  
- Number of Products  
- Customer Activity  

---

### Final Model Selection
Logistic Regression was selected due to its higher recall, ensuring fewer missed churn customers, which is critical for business impact.

---

### Business Impact
- Enables early identification of at-risk customers  
- Supports targeted retention strategies  
- Reduces revenue loss due to churn  

---

### Deployment Concept
- Model can be integrated into CRM systems  
- Used for real-time or batch churn prediction  
- Supports proactive customer engagement  

---

### How to Run

### 1. Install dependencies
pip install -r requirements.txt


### 2. Run prediction
cd app
python main.py

---

### Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  

---

#git rm -r --cached .venv## Key Learning
Model performance should be aligned with business objectives. In churn prediction, recall is more important than accuracy, as missing a churn customer has higher cost than false alarms.
