# explain.py
import shap
import pandas as pd
import numpy as np
from models import preprocessor, model  # Load model and preprocessor from models/__init__.

# Human-readable feature name mapping (customize as needed)
FEATURE_LABELS = {
    'num__tenure': 'Tenure',
    'num__MonthlyCharges': 'Monthly Charges',
    'num__TotalCharges': 'Total Charges',
    'cat__gender_Male': 'Gender: Male',
    'cat__Partner_Yes': 'Has Partner',
    'cat__Dependents_Yes': 'Has Dependents',
    'cat__PhoneService_Yes': 'Has Phone Service',
    'cat__MultipleLines_Yes': 'Multiple Lines',
    'cat__MultipleLines_No phone service': 'No Phone Service',
    'cat__InternetService_Fiber optic': 'Fiber Optic Internet',
    'cat__InternetService_No': 'No Internet Service',
    'cat__OnlineSecurity_Yes': 'Has Online Security',
    'cat__OnlineSecurity_No internet service': 'No Internet Service',
    'cat__OnlineBackup_Yes': 'Has Online Backup',
    'cat__DeviceProtection_Yes': 'Has Device Protection',
    'cat__TechSupport_Yes': 'Has Tech Support',
    'cat__StreamingTV_Yes': 'Streams TV',
    'cat__StreamingMovies_Yes': 'Streams Movies',
    'cat__Contract_One year': 'One-year Contract',
    'cat__Contract_Two year': 'Two-year Contract',
    'cat__PaperlessBilling_Yes': 'Paperless Billing',
    'cat__PaymentMethod_Credit card (automatic)': 'Credit Card (auto)',
    'cat__PaymentMethod_Electronic check': 'Electronic Check',
    'cat__PaymentMethod_Mailed check': 'Mailed Check',
    # Add any other one-hot names you see in your output
}

def get_top_reasons(customer_df, top_n=5):
    """
    Get top SHAP reasons for a single customer.
    Returns list of dicts with clean labels.
    """
    # Preprocess
    customer_transformed = preprocessor.transform(customer_df)

    # Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer_transformed)

    # For binary classification, use class 1 (churn)
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap_for_customer = shap_values[0] if len(shap_values.shape) == 2 else shap_values

    # Feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP DataFrame
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_for_customer
    })

    # Clean / human-readable names
    shap_df['clean_name'] = shap_df['feature'].map(FEATURE_LABELS).fillna(shap_df['feature'])

    # Sort by absolute impact
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(top_n)

    results = []
    for _, row in shap_df.iterrows():
        shap_val = row['shap_value']
        label = row['clean_name']

        # Determine direction and impact
        if shap_val > 0.1:
            direction = '↑'
            impact = 'High'
            color = '#ef4444'  # red
        elif shap_val > 0.05:
            direction = '↑'
            impact = 'Medium'
            color = '#f59e0b'  # orange
        elif shap_val < -0.1:
            direction = '↓'
            impact = 'High'
            color = '#10b981'  # green
        elif shap_val < -0.05:
            direction = '↓'
            impact = 'Medium'
            color = '#6ee7b7'  # light green
        else:
            direction = '→'
            impact = 'Low'
            color = '#6b7280'  # gray

        results.append({
            'direction': direction,
            'label': label,
            'impact': impact,
            'color': color,
            'shap_value': round(shap_val, 3)
        })

    return results