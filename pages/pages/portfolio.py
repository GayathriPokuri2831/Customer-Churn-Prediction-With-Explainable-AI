# pages/portfolio.py
import streamlit as st
import pandas as pd
import numpy as np
from models import preprocessor, model  # Load model and preprocessor from models/__init__.py
# Reuse your load_data from utils.py (or copy it here if not)
from utils import load_data  # if load_data is in utils.py
from theme import init_theme
init_theme()  # Apply theme CSS


df = load_data()

if st.button("← Back to Home", type="secondary"):
    st.switch_page("app.py")

# Preprocess full dataset once (cached)
@st.cache_resource
def get_full_predictions():
    X_full = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    X_full_t = preprocessor.transform(X_full)
    probs_full = model.predict_proba(X_full_t)[:, 1] * 100
    return probs_full

probs_full = get_full_predictions()

st.title("Churn Portfolio Overview")

# Hero KPIs - 4 cards
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churn_rate = (df['Churn'] == 'Yes').mean() * 100
avg_risk = round(probs_full.mean(), 1)

expected_loss = (probs_full / 100) * df['MonthlyCharges'] * 12
revenue_at_risk_annual = round(expected_loss.sum() / 10000000, 2)


with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("Actual Churn Rate", f"{churn_rate:.1f}%")

with col3:
    st.metric("Avg Predicted Risk", f"{avg_risk:.1f}%")

with col4:
    st.metric("Revenue at Risk (annual)", f"${revenue_at_risk_annual:1f} M")

# Risk Distribution Histogram
st.subheader("Risk Distribution")
risk_buckets = pd.cut(probs_full, bins=[0, 40, 70, 100], labels=["Low (<40%)", "Medium (40-70%)", "High (>70%)"])
risk_counts = pd.Series(risk_buckets).value_counts(normalize=True) * 100

st.bar_chart(risk_counts.rename("Percentage"))

# Revenue at Risk by Contract Type
st.subheader("Revenue at Risk by Contract Type")
df['pred_risk'] = probs_full
df['expected_loss_annual'] = expected_loss
contract_risk = df.groupby('Contract')['expected_loss_annual'].sum()

st.bar_chart(contract_risk.rename("Annual Expected Revenue Loss"))

st.caption("High-level snapshot using real model predictions")