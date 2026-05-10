Churn Prediction Data Pipeline
==============================

Purpose
-------
This pipeline prepares raw customer banking data for churn prediction modeling.
The goal is to clean inconsistent data, engineer meaningful features, and convert
the dataset into a machine-learning-ready format.

PIPELINE ARCHITECTURE
=====================
RAW DATA
↓
DATA CLEANING
↓
FEATURE ENGINEERING
↓
DATA TRANSFORMATION
↓
PROCESSED DATASET
↓
MACHINE LEARNING MODEL


Pipeline Steps
--------------

1. Data Cleaning
----------------
- Remove unnecessary columns:
  - id
  - full_name
  - address

These columns do not help predict churn and may introduce noise.

- Convert date columns into datetime format:
  - created_date
  - last_active_date

This allows time-based feature engineering.

- Convert last_transaction_month into numeric format.

Although the column name suggests a date, the dataset description defines it
as a transaction/activity metric from the customer’s latest active month.

- Handle missing values:
  - Numeric columns → filled using median
  - Categorical columns → filled with "Unknown"

Median is used because financial data is often skewed and sensitive to outliers.


2. Feature Engineering
----------------------
The pipeline creates new features that better represent customer behavior.

- recency_days
  Measures how recently the customer was active.

- tenure_days
  Measures how long the customer has been with the bank.

- log_balance
- log_income
- log_last_transaction

Log transformations reduce skewness in financial data and improve model stability.

- balance_income_ratio
  Measures customer financial capacity relative to income.

- transaction_income_ratio
  Measures transaction activity relative to income.

These engineered features help the model better identify churn patterns.


3. Remove Unnecessary Features
------------------------------
- risk_score
- cluster_group
- raw datetime columns

Raw date columns are removed after extracting useful time-based information.
Most machine learning models work better with numeric features instead of
datetime objects.
