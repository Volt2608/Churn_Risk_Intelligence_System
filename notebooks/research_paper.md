# Churn Risk Intelligence System for Retail Banking


|**Jyotika Bhatia**  | **Saiprasad Kagne**|
|---|---|
| Data Science Intern | Data Scientist |
| Noida, India | Unified Mentor, Delhi India |

## Abstract
Customer churn is a major risk for retail banking because it directly affects customer lifetime value, recurring revenue, and cross-sell potential. This project implements an end-to-end churn prediction system using customer-level banking data and aligns the implementation with product requirements for predictive intelligence, explainability, and decision support. The workflow includes data preprocessing, feature engineering, model development, threshold-based risk scoring, and a Streamlit dashboard for business users. The deployed interface supports customer-level risk estimation, scenario simulation, portfolio-level monitoring, and revenue-at-risk analysis in EUR. The final system is designed to be practical, interpretable, and suitable for retention campaign planning.

**Keywords:** Customer churn, retail banking, machine learning, risk scoring, explainable AI, Streamlit.

## 1. Introduction
In retail banking, churn prediction shifts decision-making from reactive to proactive. Instead of analyzing churn after loss happens, predictive models allow institutions to identify likely churners in advance and intervene with targeted offers, service actions, and engagement programs.

This project was developed as a PRD-driven churn intelligence solution with three priorities:
1. Predict churn probability at customer level.
2. Explain key churn drivers to build business trust.
3. Convert predictions into retention actions and revenue-risk estimates.

## 2. Problem Statement
Banks often have rich customer data but still face:
1. Limited predictive performance in churn identification.
2. Lack of quantitative probability scores for prioritization.
3. Weak explainability for business and regulatory stakeholders.

As a result, retention programs become broad, expensive, and less effective. This project addresses these gaps with an interpretable predictive system and a decision-focused dashboard.

## 3. Objectives
### 3.1 Primary Objectives
1. Predict customer churn with high practical accuracy.
2. Generate churn probability scores (0-1) and binary churn flags.
3. Identify key churn drivers through model explainability.

### 3.2 Secondary Objectives
1. Reduce false positives through threshold management.
2. Improve transparency using feature importance and SHAP/PDP support.
3. Enable scenario-based churn analysis for business planning.

## 4. Dataset Description
Source file used in project code: `data/raw/European_Bank_data.csv`

| Column | Description |
|---|---|
| CustomerId | Unique customer identifier |
| Surname | Customer surname |
| CreditScore | Customer creditworthiness |
| Geography | France, Spain, Germany |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Credit card ownership |
| IsActiveMember | Activity indicator |
| EstimatedSalary | Estimated annual salary |
| Exited | Churn indicator (target) |

Project dataset profile:
1. Total records: 10,000
2. Churn rate (`Exited=1`): 20.37%
3. Missing values in raw data: 0 (still handled safely in code pipeline)

## 5. Methodology (Step-by-Step)
## 5.1 Data Preprocessing
1. Remove non-informative fields: `CustomerId`, `Surname`, `Year`.
2. Handle missing values (pipeline-safe approach):
   - Numeric columns: median fill
   - Categorical columns: mode fill
3. One-hot encode categorical variables (`Geography`, `Gender`) with `drop_first=True`.
4. Scale numerical feature space using `StandardScaler`.

## 5.2 Feature Engineering
The following derived features are implemented in training/inference pipeline:
1. `BalanceSalaryRatio = Balance / (EstimatedSalary + 1)`
2. `ProductDensity = NumOfProducts / (Tenure + 1)`
3. `EngagementProduct = IsActiveMember * NumOfProducts`
4. `AgeTenureInteraction = Age * Tenure`

## 5.3 Train-Test Strategy
1. Stratified train-test split is used in notebooks to preserve churn distribution.
2. Optional cross-validation is included through `StratifiedKFold`.

## 5.4 Model Development
Models covered in project workflow:
1. Logistic Regression (baseline and final deployed model for interpretability).
2. Decision Tree.
3. Random Forest.
4. Gradient Boosting.
5. XGBoost (optional; used where environment supports library).

## 5.5 Model Evaluation Metrics
| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Controls false churn alarms |
| Recall | Captures actual churners |
| F1-score | Balance of precision and recall |
| ROC-AUC | Discrimination power across thresholds |

The project also uses confusion matrix outputs for operational interpretation.

## 5.6 Explainability
Explainability support is implemented through:
1. Global feature importance ranking.
2. SHAP-based importance (environment dependent with safe fallback).
3. Partial dependence plots (PDP) for top features.

This ensures the system can be explained to both technical and non-technical stakeholders.

## 6. Results Summary
Notebook experiments report that Logistic Regression provides the most practical trade-off for this use case (strong recall and low missed-churn count), supporting business preference for identifying at-risk customers early.

Representative model comparison (from project experiments):

| Model | Accuracy | Precision | Recall | F1 Score | Verdict |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.85 | 0.324 | 0.826 | 0.466 | Final deployed model |
| Random Forest | 0.826 | 0.563 | 0.661 | 0.608 | Good balance alternative |
| XGBoost (optional) | 0.842 | 0.609 | 0.624 | 0.617 | Strong optional model |
| Decision Tree | 0.786 | 0.474 | 0.499 | 0.486 | Weaker generalization |

Note: Exact values can vary slightly with environment and retraining.

## 7. Streamlit Application (PRD Modules)
The deployed application in `app/streamlit_app.py` includes:
1. Customer churn risk calculator.
2. Probability distribution and risk-threshold visualizations.
3. Feature importance dashboard.
4. What-if scenario simulator.
5. Portfolio-level analytics:
   - Risk bands
   - Geography x Gender heatmap
   - Revenue-at-risk (EUR)
   - Save-candidate prioritization
6. Executive summary export (PDF) for stakeholders.

### User Capabilities
1. Enter customer profile and account attributes.
2. Adjust engagement/product inputs and simulate risk changes.
3. Observe churn probability delta and expected revenue-loss impact.
4. Review business recommendations driven by prediction outputs.

## 8. Business Recommendations
Based on model behavior and portfolio diagnostics:
1. Prioritize customers with high expected loss and high churn probability for immediate outreach.
2. Design reactivation programs for inactive members (`IsActiveMember=0`).
3. Use cross-sell strategies for low-product customers (`NumOfProducts<=1`) to increase stickiness.
4. Apply geography-specific interventions where predicted risk concentration is highest.
5. Operate threshold tuning based on business cost, not metric score alone.

## 9. Deliverables Mapping to PRD
| PRD Deliverable | Project Status |
|---|---|
| Research paper (EDA, insights, recommendations) | Completed (this document) |
| Streamlit dashboard (live analytics) | Completed (`app/streamlit_app.py`) |
| Executive summary for stakeholders | Completed (in-app PDF export) |

## 10. Conclusion
This project reframes churn prediction as a business decision system rather than a standalone model exercise. By combining feature-engineered churn modeling, explainability modules, and an action-oriented dashboard, the solution supports proactive retention strategy with measurable business value. The final architecture is practical for interview demonstration, stakeholder communication, and iterative improvement in production-like settings.

## Appendix A: Reproducibility
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run application:
```bash
cd /Users/frolt/Desktop/churn_risk_system
streamlit run app/streamlit_app.py
```
3. Key implementation files:
   - `models/train.py`
   - `models/evaluate.py`
   - `models/predict.py`
   - `app/streamlit_app.py`
