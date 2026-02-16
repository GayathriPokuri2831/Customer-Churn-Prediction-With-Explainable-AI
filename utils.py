# utils.py
import pandas as pd# Assuming models are loaded in models/__init__.py
import streamlit as st
import numpy as np

@st.cache_data

def load_data():
    df= pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # Clean TotalCharges
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with NaN in TotalCharges (the 11 tenure=0 cases)
    df = df.dropna(subset=['TotalCharges'])
    
    return df


def get_usage_matrix(df):
    """
    Creates a 2x2 usage matrix: Tenure (New/Long) × Monthly Usage (Low/High)
    Returns data ready for rendering in Streamlit.
    """
    # Define cutoffs (feel free to adjust these later)
    tenure_cutoff = 24          # New: <=24 months, Long: >24
    charges_cutoff = df['MonthlyCharges'].median()  # Low/High split at median

    # Create categories
    df_temp = df.copy()
    df_temp['TenureGroup'] = df_temp['tenure'].apply(
        lambda x: 'New' if x <= tenure_cutoff else 'Long'
    )
    df_temp['ChargesGroup'] = df_temp['MonthlyCharges'].apply(
        lambda x: 'Low' if x <= charges_cutoff else 'High'
    )

    # Calculate percentages for each cell
    total = len(df_temp)
    matrix = {}

    for tenure_g in ['New', 'Long']:
        for charge_g in ['Low', 'High']:
            subset = df_temp[
                (df_temp['TenureGroup'] == tenure_g) &
                (df_temp['ChargesGroup'] == charge_g)
            ]

            churn_rate = (subset['Churn'] == 'Yes').mean() * 100 if len(subset) > 0 else 0

            if churn_rate < 30:
                risk, color = 'Low', '#10b981'
            elif churn_rate < 60:
                risk, color = 'Medium', '#f59e0b'
            else:
                risk, color = 'High', '#ef4444'

            matrix[(tenure_g, charge_g)] = {
                'pct': round((len(subset) / total) * 100, 1),
                'churn_rate': round(churn_rate, 1),
                'risk': risk,
                'color': color
            }


    # Fill real percentages
    for tenure_g, charge_g in matrix:
        count = len(df_temp[(df_temp['TenureGroup'] == tenure_g) & (df_temp['ChargesGroup'] == charge_g)])
        matrix[(tenure_g, charge_g)]['pct'] = round((count / total) * 100, 1) if total > 0 else 0

    return matrix, tenure_cutoff, charges_cutoff


def get_peer_comparison(df, customer_row,preprocessor=None,model=None, current_prob=None):
    """
    Personalized comparison: this customer vs peers in same tenure & charge band.
    Now includes real predicted churn risk comparison.
    """
    if preprocessor is None or model is None:
        return {"error": "Preprocessor and model must be provided"}
    if current_prob is None:
        return {"error":"Current churn probability must be provided"}
    # Bands (same as matrix)
    tenure_cutoff = 24
    charges_cutoff = df['MonthlyCharges'].median()

    # Customer's bands
    tenure_group = 'New' if customer_row['tenure'] <= tenure_cutoff else 'Long'
    charges_group = 'Low' if customer_row['MonthlyCharges'] <= charges_cutoff else 'High'

    # Filter peers
    peers = df[
        (df['tenure'].apply(lambda x: (x <= tenure_cutoff) if tenure_group == 'New' else (x > tenure_cutoff))) &
        (df['MonthlyCharges'].apply(lambda x: (x <= charges_cutoff) if charges_group == 'Low' else (x > charges_cutoff)))
    ]

    if len(peers) == 0:
        return {"error": "No similar peers found"}

    # Prepare peers for prediction (need to match preprocessor input)
    peer_rows = peers.drop(columns=['customerID', 'Churn'], errors='ignore')

    # Preprocess all peers at once
    peer_transformed = preprocessor.transform(peer_rows)

    # Predict churn prob for each peer
    peer_probs = model.predict_proba(peer_transformed)[:, 1] * 100  # class 1 = churn
    peer_avg_risk = round(peer_probs.mean(), 1)  # average predicted risk

    # Customer's risk (already calculated in app.py as churn_prob)
    # But for completeness, we can re-compute here if needed

    # Risk difference %
    if peer_avg_risk > 0:
        risk_pct = round(((current_prob - peer_avg_risk) / peer_avg_risk) * 100, 0)
    else:
        risk_pct = 0

    # Build return dict
    comparisons = {
        "peer_group": f"{tenure_group} tenure + {charges_group} charges",
        "peer_count": len(peers),
        "peer_avg_risk": peer_avg_risk,  # new
        "risk_percent": risk_pct,        # new - this is what app.py uses
        "monthly_charges": {
            "your_value": round(customer_row['MonthlyCharges'], 1),
            "peer_avg": round(peers['MonthlyCharges'].mean(), 1),
            "difference_pct": round(((customer_row['MonthlyCharges'] - peers['MonthlyCharges'].mean()) / peers['MonthlyCharges'].mean()) * 100, 1)
        },
        "tech_support": round((peers['TechSupport'] == 'Yes').mean() * 100, 1),
        "online_security": round((peers['OnlineSecurity'] == 'Yes').mean() * 100, 1),
        "contract_distribution": peers['Contract'].value_counts(normalize=True).head(2) * 100
    }
    comparisons['peer_avg_risk']=peer_avg_risk
    comparisons['risk_percent']=risk_pct

    return comparisons
def simulate_what_if(
    customer_row,
    current_prob,          # expected in 0–100 scale
    discount_pct=0,        # 0–30 from UI
    extra_data_gb=0,
    loyalty_plan=None,
    preprocessor=None,
    model=None,
    min_charge=18.8,
    max_abs_drop=25.0
):
    """
    Clean What-If Simulation (Model Faithful)

    - Uses real XGBoost probability
    - Applies business guardrails
    - Keeps math consistent (0–100 scale)
    """

    new_row = customer_row.copy()
    current_charges = float(new_row['MonthlyCharges'])

    # ==========================================================
    # 1️⃣ APPLY DISCOUNT
    # ==========================================================

    reduction_pct = discount_pct / 100
    reduction_amount = current_charges * reduction_pct

    new_monthly = max(current_charges - reduction_amount, min_charge)
    new_row['MonthlyCharges'] = new_monthly

    # Keep TotalCharges consistent
    if 'TotalCharges' in new_row and 'tenure' in new_row:
        tenure = max(float(new_row['tenure']), 1)
        new_row['TotalCharges'] = new_monthly * tenure

    # ==========================================================
    # 2️⃣ APPLY LOYALTY PLAN
    # ==========================================================
    if loyalty_plan == "12-Month Contract":
        new_row["Contract"] = "One year"
    elif loyalty_plan == "24-Month Contract":
        new_row["Contract"] = "Two year"


    # ==========================================================
    # 3️⃣ APPLY EXTRA DATA (Proxy via service value)
    # ==========================================================

    if extra_data_gb > 0:
        if "OnlineSecurity" in new_row:
            new_row["OnlineSecurity"] = "Yes"
        if "TechSupport" in new_row:
            new_row["TechSupport"] = "Yes"

    # ==========================================================
    # 4️⃣ MODEL PREDICTION
    # ==========================================================

    customer_df = pd.DataFrame([new_row])
    transformed = preprocessor.transform(customer_df)

    raw_new_prob = model.predict_proba(transformed)[0][1] * 100  # 0–100 scale
    raw_new_prob = round(raw_new_prob, 1)

    # ==========================================================
    # 5️⃣ BUSINESS GUARDRAILS (Soft, Not Overriding)
    # ==========================================================

    # Prevent price reduction from increasing risk
    if discount_pct > 0 and raw_new_prob > current_prob:
        raw_new_prob = current_prob + 0.1

    # Cap excessive drop
    max_allowed_prob = current_prob - max_abs_drop
    new_prob = max(raw_new_prob, max_allowed_prob)

    new_prob = round(new_prob, 1)

    abs_drop = round(current_prob - new_prob, 1)
    rel_drop = round((abs_drop / current_prob) * 100, 1) if current_prob > 0 else 0

    # ==========================================================
    # 6️⃣ MESSAGE
    # ==========================================================

    if abs_drop > 0:
        message = f"Risk drops by {abs_drop:.1f} pts (~{rel_drop:.1f}% improvement)"
    else:
        message = "No material improvement"

    return {
        'new_prob': new_prob,
        'abs_drop_pts': abs_drop,
        'rel_drop_pct': rel_drop,
        'message': message,
        'new_charges': round(new_monthly, 1),
        'reduction_amount': round(reduction_amount, 1),
        'loyalty_applied': loyalty_plan,
        'extra_data_gb': extra_data_gb
    }


# ────────────────────────────────────────────────
# Simulated Drift Check (runs once on app start)
# ────────────────────────────────────────────────
def check_simulated_drift(df, model, preprocessor, threshold=0.12):
    """
    Simulate drift detection: compare AUC-PR on early vs recent data.
    Returns dict: detected (bool), drop_pct, message, last_checked
    """
    if len(df) < 100:
        return {"detected": False, "drop_pct": 0, "message": "Dataset too small", "last_checked": pd.Timestamp.now().strftime("%b %d, %Y %H:%M IST")}

    # Split: first 80% "training period", last 20% "recent"
    split_idx = int(len(df) * 0.8)
    df_early = df.iloc[:split_idx]
    df_recent = df.iloc[split_idx:]

    # Prepare early & recent
    X_early = df_early.drop(columns=['customerID','Churn'],errors='ignore')
    y_early = df_early['Churn'].map({'No': 0, 'Yes': 1})
    X_recent = df_recent.drop(columns=['customerID','Churn'],errors='ignore')
    y_recent = df_recent['Churn'].map({'No': 0, 'Yes': 1})

    # Preprocess
    X_early_t = preprocessor.transform(X_early)
    X_recent_t = preprocessor.transform(X_recent)

    # Predict probs
    prob_early = model.predict_proba(X_early_t)[:, 1]
    prob_recent = model.predict_proba(X_recent_t)[:, 1]

    # Compute AUC-PR
    from sklearn.metrics import average_precision_score
    aucpr_early = average_precision_score(y_early, prob_early)
    aucpr_recent = average_precision_score(y_recent, prob_recent)

    drop_pct = aucpr_early - aucpr_recent
    detected = drop_pct > threshold

    message = (
        f"AUC-PR dropped by {drop_pct*100:.1f}% on recent data "
        f"(early: {aucpr_early:.3f} → recent: {aucpr_recent:.3f})"
        if detected else "No significant degradation detected"
    )

    return {
        "detected": detected,
        "drop_pct": round(drop_pct * 100, 1),
        "message": message,
        "last_checked": pd.Timestamp.now().strftime("%b %d, %Y %H:%M IST")
    }

