# pages/priority.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data
from models import preprocessor, model  # Load model and preprocessor from models/__init__.py
from theme import init_theme
init_theme()  # Apply theme CSS


df = load_data()

if st.button("← Back to Home", type="secondary"):
    st.switch_page("app.py")

st.title("Retention Priority List")

# Preprocess full dataset once (cached)
@st.cache_data
def get_all_predictions():
    X = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    X_t = preprocessor.transform(X)
    probs = model.predict_proba(X_t)[:, 1] * 100
    df['Predicted Risk %'] = np.round(probs, 1)
    return df

df_risk = get_all_predictions()

# Filters (sidebar)
st.sidebar.header("Filters")

min_risk = st.sidebar.slider("Minimum Risk %", 0, 100, 50)
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df_risk['Contract'].unique(),
    default=df_risk['Contract'].unique()
)
tenure_min, tenure_max = st.sidebar.slider(
    "Tenure Range (months)",
    0, int(df_risk['tenure'].max()),
    (0, int(df_risk['tenure'].max()))
)

# Apply filters
filtered = df_risk[
    (df_risk['Predicted Risk %'] >= min_risk) &
    (df_risk['Contract'].isin(contract_filter)) &
    (df_risk['tenure'].between(tenure_min, tenure_max))
]

# Sortable table (top 200 high-risk)
st.subheader(f"Top Priority Customers ({len(filtered):,} found)")

# Select columns to show
columns_to_show = [
    'customerID',
    'Predicted Risk %',
    'MonthlyCharges',
    'tenure',
    'Contract',
    'TechSupport',
    'Churn'
]

st.dataframe(
    filtered[columns_to_show]
        .sort_values('Predicted Risk %', ascending=False)
        .head(200)
        .reset_index(drop=True),
        width='stretch',
        hide_index=True
)

# Summary cards (top)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("High-Risk Customers", f"{len(filtered[filtered['Predicted Risk %'] >= 70]):,}")

with col2:
    st.metric("Avg Risk in List", f"{filtered['Predicted Risk %'].mean():.1f}%")

with col3:
    revenue_at_risk = 0
    if len(filtered) > 0:
        revenue_at_risk = (
            filtered['MonthlyCharges'] *
            filtered['Predicted Risk %'] / 100
        ).sum()


    st.metric("Est. Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

st.caption("Sortable table — click column headers to sort. Filtered to high-priority only.")