# A Project Report on Customer Churn Prediction using Machine Learning

**Jyotika Bhatia**  
Data Science Intern  
Noida, India  

**Saiprasad Kagne**  
Data Scientist  
Unified Mentor, Delhi, India  

---

## Abstract
Customer churn is a major risk in retail banking as it directly impacts customer lifetime value and revenue. This project presents an end-to-end churn prediction system using customer-level banking data, with a focus on predictive modeling, interpretability, and business decision support. The workflow includes data preprocessing, feature engineering, model development, and a Streamlit dashboard for interactive usage. The system supports customer-level risk estimation, scenario simulation, and portfolio-level monitoring. The final solution is designed to be practical, interpretable, and useful for retention strategy planning.

---

## Keywords
Customer churn, retail banking, machine learning, risk scoring, interpretability, Streamlit

---

## 1. Introduction
In retail banking, churn prediction enables proactive decision-making. Instead of analyzing churn after it occurs, predictive models help identify customers who are likely to leave and allow timely intervention.

This project focuses on:
- Predicting churn probability at the customer level  
- Explaining key drivers behind churn  
- Supporting business decisions through a simple analytical system  

---

## 2. Problem Statement
Banks often face:
- Limited ability to accurately identify churners  
- Lack of probability-based prioritization  
- Low transparency in model predictions  

As a result, retention strategies become inefficient and costly. This project addresses these challenges using a data-driven and interpretable churn prediction system.

---

## 3. Objectives

### Primary Objectives
- Predict customer churn with practical accuracy  
- Generate churn probability scores (0–1)  
- Identify key churn drivers  

### Secondary Objectives
- Enable threshold-based decision-making  
- Provide interpretable insights for business users  
- Support scenario-based analysis  

---

## 4. Dataset Description
The dataset represents customer information from a retail banking context.

### Key Features
- CreditScore, Geography, Gender, Age  
- Tenure, Balance, NumOfProducts  
- HasCrCard, IsActiveMember, EstimatedSalary  

### Target Variable
- Exited (1 = churn, 0 = retained)

### Dataset Profile
- Total records: 10,000  
- Churn rate: ~20.37%  
- Missing values: Not present (handled safely in pipeline)

---

## 5. Methodology

### 5.1 Data Preprocessing
- Removed non-informative features (CustomerId, Surname)  
- Handled missing values (median for numerical, mode for categorical)  
- One-hot encoding for categorical variables  
- Feature scaling using StandardScaler  

### 5.2 Feature Engineering
Derived features:
- BalanceSalaryRatio  
- ProductDensity  
- EngagementProduct  
- AgeTenureInteraction  

### 5.3 Train-Test Strategy
- Stratified train-test split  
- Optional cross-validation using StratifiedKFold  

### 5.4 Model Development
Models implemented:
- Logistic Regression (baseline and final model)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost (optional, environment-dependent)

### 5.5 Model Evaluation
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

Higher recall is prioritized to minimize missed churners, as the cost of losing a customer is typically higher than the cost of unnecessary retention efforts.

### 5.6 Interpretability
Interpretability is supported through:
- Feature importance analysis  
- Basic model-driven insights for business understanding  

---

## 6. Results Summary

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|----------|--------|----------|
| Logistic Regression | 0.85     | 0.324    | 0.826  | 0.466    |
| Random Forest       | 0.826    | 0.563    | 0.661  | 0.608    |
| XGBoost (optional)  | 0.842    | 0.609    | 0.624  | 0.617    |
| Decision Tree       | 0.786    | 0.474    | 0.499  | 0.486    |

Logistic Regression is selected as the final model due to its strong recall and interpretability, making it suitable for identifying at-risk customers in a business context.

---

## 7. Streamlit Application

The deployed application includes:
- Customer churn prediction interface  
- Probability-based risk scoring  
- Scenario simulation (what-if analysis)  
- Feature importance insights  
- Portfolio-level analytics  

### User Capabilities   
- Input customer data  
- Simulate changes in features  
- Observe churn probability changes  
- Support retention decision-making  

---

## 8. Business Recommendations
- Prioritize high-risk customers for targeted retention campaigns  
- Focus on inactive users for re-engagement  
- Increase product usage to improve customer retention  
- Apply threshold tuning based on business cost considerations  

---

## 9. Conclusion
This project demonstrates how machine learning can be applied to solve a real-world business problem. By combining predictive modeling with an interactive dashboard, the system enables proactive and data-driven retention strategies. The approach is practical, interpretable, and suitable for real-world applications.

---

## 10. Reproducibility

### Install dependencies
```bash
pip install -r requirements.txt