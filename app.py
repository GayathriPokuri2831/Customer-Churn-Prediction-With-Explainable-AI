# app.py - Home / Navigation Page

import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Churn Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Churn Retention Dashboard")

# ---- GLOBAL TILE CSS ----
st.markdown("""
<style>

/* Base tile style */
.tile-link {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 150px;
    font-size: 22px;
    font-weight: 600;
    border-radius: 14px;
    text-decoration: none !important;
    color: white !important;
    transition: all 0.25s ease;
    margin-bottom: 1.5rem;
    letter-spacing: 0.3px;
            
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}

/* Hover animation */
.tile-link:hover {
    transform: translateY(-4px);
    box-shadow: 0px 12px 28px rgba(0,0,0,0.45);
    filter: brightness(1.08);
}

/* Gradient themes */
.blue { background: linear-gradient(135deg, #2563EB, #1E3A8A); }
.green { background: linear-gradient(135deg, #10B981, #065F46); }
.purple { background: linear-gradient(135deg, #8B5CF6, #4C1D95); }
.red { background: linear-gradient(135deg, #EF4444, #7F1D1D); }
.cyan{ background: linear-gradient(135deg, #14B8A6, #134E4A); }

</style>
""", unsafe_allow_html=True)

# ---- ROW 1 ----
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <a href="/portfolio" target="_self" class="tile-link blue">
            Portfolio Overview
        </a>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <a href="/priority" target="_self" class="tile-link red">
            Priority Customer Ranking
        </a>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <a href="/cohorts" target="_self" class="tile-link purple">
            Cohort & Segment Trends
        </a>
        """,
        unsafe_allow_html=True
    )
col4, col5= st.columns(2)
with col4:
    st.markdown(
        """
        <a href="/dashboard" target="_self" class="tile-link cyan">
            Customer Search & Analysis
        </a>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        """
        <a href="/whatif" target="_self" class="tile-link green">
            Retention Strategy Simulator
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Built with XGBoost + SHAP | Churn Prediction & Retention")
