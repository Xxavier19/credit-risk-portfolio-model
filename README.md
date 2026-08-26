# Credit Risk Portfolio Modeling & Stress Testing

An end-to-end credit risk modeling project that uses machine learning to estimate borrower probability of default, quantify portfolio expected loss, perform stress testing, and provide interactive borrower-level risk analysis through a Streamlit dashboard.

## Project Overview

This project analyzes Lending Club loan data to build a machine learning-based credit risk framework.

The project combines traditional credit risk concepts with machine learning to:

- Predict borrower probability of default (PD)
- Compare multiple classification models
- Estimate expected credit loss using PD, LGD, and EAD
- Segment a loan portfolio by credit risk
- Perform portfolio stress testing
- Analyze model behavior using SHAP
- Identify high-risk borrowers and their risk drivers
- Deploy the final model through an interactive Streamlit dashboard

## Machine Learning Models

Three classification models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The final model selected was **XGBoost**, which provided the strongest overall combination of discrimination and classification performance.

| Model | ROC-AUC |
|------------------------------
| Logistic Regression | 0.736 |
| Random Forest | 0.714 |
| XGBoost | 0.736 |

Model tuning was also performed to compare optimized versions of each algorithm.

## Credit Risk Framework

The model's predicted probability is interpreted as the borrower's **Probability of Default (PD)**.

Portfolio expected loss is calculated using:

**Expected Loss = PD × LGD × EAD**

Where:

- **PD** = Probability of Default
- **LGD** = Loss Given Default
- **EAD** = Exposure at Default

A 60% LGD assumption is used in the portfolio analysis.

## Portfolio Analysis

The Streamlit dashboard calculates:

- Total portfolio exposure
- Average predicted PD
- Total expected loss
- Expected loss rate
- Risk segmentation
- Exposure by risk level
- Expected loss concentration by risk level

Loans are segmented into four risk categories:

- Low Risk
- Moderate Risk
- High Risk
- Very High Risk

## Stress Testing

The dashboard includes an interactive stress-testing framework that allows users to increase portfolio PD assumptions.

The stress test dynamically recalculates:

- Stressed expected loss
- Dollar increase in expected loss
- Percentage increase in expected loss
- Baseline vs. stressed portfolio losses

This provides a simplified example of how deterioration in borrower credit quality can affect portfolio-level losses.

## Model Explainability

SHAP analysis was used in the modeling notebook to understand both global and borrower-level model behavior.

The analysis identifies which borrower characteristics have the greatest influence on XGBoost predictions and demonstrates why individual borrowers may receive elevated default probabilities.

## Interactive Borrower Risk Analysis

The Streamlit application allows users to enter borrower characteristics including:

- Loan amount
- Interest rate
- Loan term
- Income
- Debt-to-income ratio
- FICO score
- Revolving utilization
- Loan grade and sub-grade
- Employment length
- Home ownership
- Verification status
- Loan purpose

The model then returns:

- Predicted probability of default
- Risk classification
- Estimated expected loss
- Comparison of key borrower characteristics with portfolio averages

## Technology Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Altair
- Streamlit
- Joblib
- Jupyter / Google Colab
- VS Code

## Project Structure

```text
credit-risk-portfolio-model/
│
├── app.py
├── Credit_Risk_Portfolio_Model_Final.ipynb
├── lending_club_100k.csv
├── credit_risk_preprocessor.pkl
├── xgb_credit_risk_model.json
├── requirements.txt
├── README.md
└── .gitignore
