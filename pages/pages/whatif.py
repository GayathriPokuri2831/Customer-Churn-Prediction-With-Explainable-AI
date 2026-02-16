# pages/whatif.py
import streamlit as st
import pandas as pd
from utils import simulate_what_if
from explain import get_top_reasons
from theme import init_theme
from models import preprocessor, model  # Load model and preprocessor from models/__init__.py   
init_theme()  # Apply theme CSS 


st.set_page_config(layout="wide")

st.title("Retention Strategy Simulator")

if st.button("← Back to Dashboard", type="secondary"):
    st.switch_page("pages/dashboard.py")

# ==============================
# Load Customer from session state
# ==============================
if "current_row" not in st.session_state or "current_churn_prob" not in st.session_state:
    st.warning("Please load a customer first from Dashboard.")
    st.stop()

row = st.session_state["current_row"]
churn_prob = st.session_state["current_churn_prob"]

# ==============================
# Sidebar - Retention Strategy
# ==============================
st.sidebar.header("🎯 Retention Strategy")

discount_pct = st.sidebar.slider(
    "Offer Discount (%)",
    min_value=0,
    max_value=30,
    value=10,
    step=5
)

extra_data_gb = st.sidebar.selectbox(
    "Free Data Add-on",
    options=[0, 2, 5, 10],
    format_func=lambda x: f"{x} GB" if x > 0 else "None"
)

loyalty_plan = st.sidebar.selectbox(
    "Contract Upgrade",
    ["No Change", "12-Month Contract", "24-Month Contract"]
)

# ==============================
# Default Simulation Result (show current state immediately)
# ==============================
sim_result = {
    'new_prob': churn_prob,
    'abs_drop_pts': 0,
    'message': "Current state (no changes applied)",
    'new_charges': float(row['MonthlyCharges']),
    'reduction_amount': 0
}

# ==============================
# Run Simulation on button click
# ==============================
if st.sidebar.button("Simulate Strategy", type="primary"):
    sim_result = simulate_what_if(
        customer_row=row,
        current_prob=churn_prob,
        discount_pct=discount_pct,
        extra_data_gb=extra_data_gb,
        loyalty_plan=loyalty_plan ,
        preprocessor=preprocessor,
        model=model
    )

# ==============================
# Risk Comparison (always shown)
# ==============================
delta = max(churn_prob - sim_result['new_prob'], 0)

col1, col2 = st.columns(2)

with col1:
    st.metric("Current Churn Risk", f"{churn_prob:.1f}%")

with col2:
    st.metric(
        "New Churn Risk",
        f"{sim_result['new_prob']:.1f}%",
        delta=f"-{delta:.1f} pts" if delta > 0 else "0 pts",
        delta_color="normal"
    )

# ==============================
# Financial Impact
# ==============================
st.subheader("Financial Impact")

monthly_charges = float(row["MonthlyCharges"])
annual_revenue = monthly_charges * 12

expected_revenue_saved = (delta / 100) * annual_revenue

discount_cost = sim_result['reduction_amount'] * 12

data_cost_per_gb_month = 10
data_cost = extra_data_gb * data_cost_per_gb_month * 12

loyalty_cost = 100 if loyalty_plan != "No Change" else 0

intervention_cost = discount_cost + data_cost + loyalty_cost

net_benefit = expected_revenue_saved - intervention_cost

roi = net_benefit / intervention_cost if intervention_cost > 0 else None

colA, colB, colC, colD = st.columns(4)

colA.metric("Annual Revenue", f"${annual_revenue:,.0f}")
colB.metric("Expected Revenue Saved", f"${expected_revenue_saved:,.0f}")
colC.metric("Intervention Cost", f"${intervention_cost:,.0f}")
colD.metric("Net Benefit", f"${net_benefit:,.0f}")

if roi is not None:
    if roi > 1:
        st.success(f"Estimated ROI: {roi:.2f}x (Profitable)")
    else:
        st.warning(f"Estimated ROI: {roi:.2f}x (Not profitable)")

# ==============================
# Strategy Summary
# ==============================
st.subheader("Strategy Summary")

actions = []
if discount_pct > 0:
    actions.append(f"{discount_pct}% Discount")
if extra_data_gb > 0:
    actions.append(f"{extra_data_gb} GB Free Data")
if loyalty_plan != "No Change":
    actions.append(f"{loyalty_plan}")

if actions:
    st.markdown(f"**Applied Strategy:** {', '.join(actions)}")
    st.markdown(f"**New Estimated Risk:** {sim_result['new_prob']:.1f}%")
    st.markdown(f"**Risk Reduction:** {sim_result['message']}")
else:
    st.info("No strategy selected yet.")