import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import joblib

import model_statistics

# Phase 1: Upload user files
st.title('Loan Credibility Prediction')
st.header('Phase 1: Upload Required Files')

# Define schema
schemas = {
    "banking_transactions.csv": """
        customer_id,                  # Unique identifier for the customer
        transaction_date,            # Date of transaction (YYYY-MM-DD)
        transaction_amount,          # Amount in currency
        transaction_type,            # Credit/Debit
        transaction_category,        # E.g., Groceries, Utilities, Entertainment
        payment_method,             # Cash/Card/UPI/NetBanking etc.
        merchant_category_code,      # Standard merchant category code
        balance_after_transaction,   # Account balance after this transaction
        transaction_status,         # Success/Failed/Bounced
        penalty_if_any              # Any penalties applied
    """,
    "account_summary.csv": """
        customer_id,                  # Unique identifier
        account_opening_date,        # YYYY-MM-DD
        account_type,               # Savings/Current/Fixed Deposit
        primary_bank_flag,          # Is this the primary bank account?
        avg_monthly_balance_6m,     # Average balance last 6 months
        avg_monthly_balance_12m,    # Average balance last 12 months
        min_balance_breach_count,   # Number of minimum balance breaches
        digital_banking_score,      # Score based on digital banking usage
        standing_instructions_count # Number of regular payment instructions
    """,
    "credit_history.csv": """
        customer_id,                  # Unique identifier
        credit_score,               # Credit bureau score
        total_loans_active,         # Number of active loans
        total_credit_limit,         # Total credit limit across all products
        total_credit_exposure,      # Current outstanding across all loans
        loan_id,                    # Unique loan identifier
        loan_type,                  # Personal/Home/Auto/Education
        loan_amount,                # Original loan amount
        loan_start_date,            # YYYY-MM-DD
        loan_end_date,              # YYYY-MM-DD
        emi_amount,                 # Monthly EMI amount
        payments_missed_count,      # Number of missed payments
        days_past_due              # Current days past due if any
    """,
    "income_data.csv": """
        customer_id,                  # Unique identifier
        tax_year,                   # Financial year
        annual_income,              # Gross annual income
        income_source,              # Salary/Business/Investment
        employer_name,              # Current employer
        employment_start_date,      # YYYY-MM-DD
        employment_sector,          # Industry sector
        profession_category,        # Job category
        monthly_salary_credits,     # Last 12 months of salary credits
        variable_pay_amount,        # Annual bonus/variable component
        other_income_sources      # Additional income sources
    """,
    "collateral_details.csv": """
        customer_id,                  # Unique identifier
        collateral_id,              # Unique collateral identifier
        collateral_type,            # Property/Gold/Shares etc.
        collateral_value,           # Current market value
        valuation_date,            # Date of last valuation
        asset_quality_grade,        # Asset quality rating
        location_if_property,       # Location details if property
        ownership_type,            # Single/Joint
        encumbrance_details,       # Any existing liens
        liquidity_score           # Ease of liquidation score
    """,
    "insurance_details.csv": """
        customer_id,                  # Unique identifier
        policy_id,                  # Insurance policy ID
        insurance_type,            # Life/Health/Property
        coverage_amount,           # Total coverage amount
        premium_amount,            # Regular premium amount
        premium_frequency,         # Monthly/Quarterly/Annual
        last_premium_date,         # Date of last premium
        missed_premium_count,      # Number of missed premiums
        claim_history,            # Any claims made
        policy_status             # Active/Lapsed/Paid-up
    """,
    "social_factors.csv": """
        customer_id,                  # Unique identifier
        criminal_record_flag,        # Yes/No
        legal_penalties_amount,      # Total legal penalties if any
        education_level,            # Highest education
        residential_status,         # Own/Rent
        years_at_current_address,   # Duration at current address
        family_size,               # Number of dependents
        spouse_income,             # If applicable
        social_score,              # Derived social stability score
        peer_group_category       # Income-profession peer group
    """,
    "risk_labels.csv": """
        customer_id,                  # Unique identifier
        application_date,           # Date of loan application
        risk_score,                # Assigned risk score
        approval_status,           # Approved/Rejected
        default_flag,              # If loan defaulted
        default_date,              # Date of default if any
        recovery_amount,           # Amount recovered if defaulted
        loss_given_default        # Actual loss amount
    """
}

# Streamlit UI components
st.title("Upload Files and View Schemas")

# File upload and schema display for each CSV file
uploaded_account_summary = st.file_uploader("Upload Account Summary CSV", type=["csv"])
if uploaded_account_summary:
    with st.expander("View Schema for Account Summary CSV"):
        st.code(schemas["account_summary.csv"])

uploaded_banking_transactions = st.file_uploader("Upload Banking Transactions CSV", type=["csv"])
if uploaded_banking_transactions:
    with st.expander("View Schema for Banking Transactions CSV"):
        st.code(schemas["banking_transactions.csv"])

uploaded_collateral_details = st.file_uploader("Upload Collateral Details CSV", type=["csv"])
if uploaded_collateral_details:
    with st.expander("View Schema for Collateral Details CSV"):
        st.code(schemas["collateral_details.csv"])

uploaded_credit_history = st.file_uploader("Upload Credit History CSV", type=["csv"])
if uploaded_credit_history:
    with st.expander("View Schema for Credit History CSV"):
        st.code(schemas["credit_history.csv"])

uploaded_income_data = st.file_uploader("Upload Income Data CSV", type=["csv"])
if uploaded_income_data:
    with st.expander("View Schema for Income Data CSV"):
        st.code(schemas["income_data.csv"])

uploaded_insurance_details = st.file_uploader("Upload Insurance Details CSV", type=["csv"])
if uploaded_insurance_details:
    with st.expander("View Schema for Insurance Details CSV"):
        st.code(schemas["insurance_details.csv"])

uploaded_risk_labels = st.file_uploader("Upload Risk Labels CSV", type=["csv"])
if uploaded_risk_labels:
    with st.expander("View Schema for Risk Labels CSV"):
        st.code(schemas["risk_labels.csv"])

uploaded_social_factors = st.file_uploader("Upload Social Factors CSV", type=["csv"])
if uploaded_social_factors:
    with st.expander("View Schema for Social Factors CSV"):
        st.code(schemas["social_factors.csv"])

if uploaded_account_summary and uploaded_banking_transactions and uploaded_collateral_details and uploaded_credit_history and uploaded_income_data and uploaded_insurance_details and uploaded_risk_labels and uploaded_social_factors:
    st.success("All files uploaded successfully! Proceed to the next phase.")

    # Phase 2: Train the model
    st.header('Phase 2: Train the Model')
    st.text("Training the model... please wait.")
    
    # Read CSVs
    account_summary = pd.read_csv(uploaded_account_summary)
    banking_transactions = pd.read_csv(uploaded_banking_transactions)
    collateral_details = pd.read_csv(uploaded_collateral_details)
    credit_history = pd.read_csv(uploaded_credit_history)
    income_data = pd.read_csv(uploaded_income_data)
    insurance_details = pd.read_csv(uploaded_insurance_details)
    risk_labels = pd.read_csv(uploaded_risk_labels)
    social_factors = pd.read_csv(uploaded_social_factors)

    # Merge datasets on customer_id
    merged_data = account_summary.merge(banking_transactions, on='customer_id', how='left')
    merged_data = merged_data.merge(collateral_details, on='customer_id', how='left')
    merged_data = merged_data.merge(credit_history, on='customer_id', how='left')
    merged_data = merged_data.merge(income_data, on='customer_id', how='left')
    merged_data = merged_data.merge(insurance_details, on='customer_id', how='left')
    merged_data = merged_data.merge(risk_labels, on='customer_id', how='left')
    merged_data = merged_data.merge(social_factors, on='customer_id', how='left')

    # Handle missing values
    merged_data.fillna(0, inplace=True)

    # Feature engineering
    merged_data['loan_to_income_ratio'] = merged_data['loan_amount'] / (merged_data['annual_income'] + 1)
    merged_data['collateral_to_loan_ratio'] = merged_data['collateral_value'] / (merged_data['loan_amount'] + 1)
    merged_data['monthly_emi_to_income_ratio'] = merged_data['emi_amount'] / (merged_data['annual_income'] / 12 + 1)

    # Target variable: Approval status (1 for approved, 0 for rejected)
    merged_data['approval_status'] = merged_data['approval_status'].apply(lambda x: 1 if x == 'Approved' else 0)

    # Feature selection
    features = [
        'avg_monthly_balance_6m', 'avg_monthly_balance_12m', 'credit_score',
        'loan_to_income_ratio', 'collateral_to_loan_ratio', 'monthly_emi_to_income_ratio',
        'risk_score', 'social_score', 'education_level', 'criminal_record_flag'
    ]
    X = merged_data[features]
    y = merged_data['approval_status']

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    # Save the model for future use
    joblib.dump(model, "loan_credibility_model.pkl")

    st.success("Model trained successfully!")
    st.write("Accuracy: ", model_statistics.get_accuracy())
    st.write("Precision: ", model_statistics.get_precision())
    st.write("Time taken: #TODO")
    # st.text_area("Classification Report", classification_report_str, height=300)

    # Phase 3: User input for evaluation
    st.header('Phase 3: Evaluate the Model')
    
    new_customer_data = {
        'avg_monthly_balance_6m': st.number_input('Average Monthly Balance 6M', min_value=0, step=100),
        'avg_monthly_balance_12m': st.number_input('Average Monthly Balance 12M', min_value=0, step=100),
        'credit_score': st.number_input('Credit Score', min_value=0, max_value=1000, step=10),
        'loan_to_income_ratio': st.number_input('Loan to Income Ratio', min_value=0.0, max_value=1.0, step=0.01),
        'collateral_to_loan_ratio': st.number_input('Collateral to Loan Ratio', min_value=0.0, step=0.01),
        'monthly_emi_to_income_ratio': st.number_input('Monthly EMI to Income Ratio', min_value=0.0, step=0.01),
        'risk_score': st.number_input('Risk Score', min_value=0, step=1),
        'social_score': st.number_input('Social Score', min_value=0, step=1),
        'education_level': st.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'PhD']),
        'criminal_record_flag': st.selectbox('Criminal Record Flag', [0, 1])
    }

    # Prepare input data for prediction
    new_customer_df = pd.DataFrame([new_customer_data])

    # Process the input data to match model features
    new_customer_processed = pd.get_dummies(new_customer_df, drop_first=True).reindex(columns=X.columns, fill_value=0)

    # Load the model
    model = joblib.load("loan_credibility_model.pkl")

    # Predict using the trained model
    prediction = model.predict(new_customer_processed)
    prediction_probability = model.predict_proba(new_customer_processed)[0][1]

    st.write("Prediction (1 = Approved, 0 = Rejected):", prediction[0])
    st.write("Probability of Approval:", prediction_probability)

    # Download button for trained model
    with open("loan_credibility_model.pkl", "rb") as file:
        st.download_button(label="Download Trained Model", data=file, file_name="loan_credibility_model.pkl", mime="application/octet-stream")

else:
    st.warning("Please upload all the required files to proceed.")
