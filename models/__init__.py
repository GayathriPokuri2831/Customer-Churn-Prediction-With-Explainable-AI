import joblib
import streamlit as st

@st.cache_resource
def load_model_and_preprocessor():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/xgb_model.pkl")
    return preprocessor, model

preprocessor, model = load_model_and_preprocessor()