import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION & LOADING
# ==========================================
st.set_page_config(
    page_title="Credit Risk Predictor", 
    page_icon="asset/logo.png", 
    layout="wide"
)

# Load the model and median dictionary
@st.cache_resource
def load_assets():
    model = joblib.load('saved_model/lgbm_best_model.pkl')
    medians = joblib.load('saved_model/train_medians.pkl')
    return model, medians

model, medians = load_assets()

# ==========================================
# APP HEADER & SIDEBAR
# ==========================================
st.title("Home Credit Default Risk Predictor")
st.markdown("""
This application predicts the probability of a client defaulting on a loan using a **LightGBM** machine learning model. 
To keep the experience seamless, we only require the top 8 most critical predictive features. All other background financial data is automatically imputed using historical median baselines.
""")
st.divider()

with st.sidebar:
    st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <h3 style="margin-bottom: 0px;">
        <span class="material-icons" style="vertical-align: text-bottom;">settings</span> 
        Bank Risk Controls
    </h3>
    """, unsafe_allow_html=True)
    
    custom_threshold = st.slider(
        "Decision Threshold", 
        min_value=0.10, 
        max_value=0.90, 
        value=0.50, 
        step=0.05,
        help="Adjust the bank's risk appetite. Higher thresholds mean the bank is more strict and approves fewer loans."
    )
    st.info(f"Current Policy: Reject any application with > {custom_threshold*100:.0f}% default probability.")
# ==========================================
# MAIN DASHBOARD LAYOUT (LEFT & RIGHT PANELS)
# ==========================================
left_panel, right_panel = st.columns([1.2, 1], gap="large")

# ------------------------------------------
# LEFT PANEL: INPUT FORM
# ------------------------------------------
with left_panel:
    st.subheader("Client Financial Profile")
    
    with st.container(height=400, border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Application Data**")
            age_years = st.number_input("Client Age (Years)", min_value=18, max_value=100, value=35)
            amt_income_total = st.number_input("Annual Income ($)", min_value=1000, max_value=10000000, value=10000, step=5000)
            amt_credit = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=10000000, value=10000, step=5000)
            amt_annuity = st.number_input("Proposed Monthly Payment ($)", min_value=100, max_value=10000000, value=1000, step=100)
            amt_goods_price = st.number_input("Price of Goods/Asset ($)", min_value=1000, max_value=10000000, value=100000, step=5000)

        with col2:
            st.markdown("**External Bureau Data**")
            ext_source_2 = st.slider("External Credit Score 2", min_value=0.0, max_value=1.0, value=0.5)
            ext_source_3 = st.slider("External Credit Score 3", min_value=0.0, max_value=1.0, value=0.5)

        st.markdown("---")
        st.markdown("**Historical Debt Calculation**")
        
        col3, col4 = st.columns(2)
        with col3:
            bureau_total_credit = st.number_input("Total Past Credit Limit ($)", min_value=0, value=50000, step=5000)
        with col4:
            bureau_total_debt = st.number_input("Total Outstanding Past Debt ($)", min_value=0, value=15000, step=5000)

        days_birth = age_years * -365
        if bureau_total_credit > 0:
            bureau_debt_ratio = bureau_total_debt / bureau_total_credit
        else:
            bureau_debt_ratio = 0.0

        st.caption(f"*Auto-Calculated Feature:* The model is using a Debt-to-Credit Ratio of **{bureau_debt_ratio:.2f}**")
        
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict Default Risk", type="primary", use_container_width=True)

# ------------------------------------------
# RIGHT PANEL: PREDICTION RESULTS
# ------------------------------------------
with right_panel:
    st.subheader("Prediction Results")
    
    with st.container(height=400, border=True):
        if predict_button:
            user_data = medians.copy()
            
            user_data['DAYS_BIRTH'] = days_birth
            user_data['AMT_INCOME_TOTAL'] = amt_income_total
            user_data['AMT_CREDIT'] = amt_credit
            user_data['AMT_ANNUITY'] = amt_annuity
            user_data['AMT_GOODS_PRICE'] = amt_goods_price 
            user_data['BUREAU_DEBT_RATIO'] = bureau_debt_ratio
            user_data['EXT_SOURCE_2'] = ext_source_2
            user_data['EXT_SOURCE_3'] = ext_source_3
            
            input_df = pd.DataFrame([user_data])
            
            risk_probability = model.predict_proba(input_df)[0][1]
            
            st.metric(label="Calculated Default Risk", value=f"{risk_probability * 100:.2f}%")
            
            if risk_probability >= custom_threshold:
                st.error("**HIGH RISK: LOAN REJECTED**")
                st.write(f"This applicant exceeds the bank's maximum risk threshold of {custom_threshold*100:.0f}%.")
            else:
                st.success("**LOW RISK: LOAN APPROVED**")
                st.write(f"This applicant is within acceptable risk parameters.")
                
            st.progress(float(risk_probability), text="Risk Threshold Meter")
            
            # ==========================================
            # LOCAL SHAP EXPLAINER
            # ==========================================
            st.divider()
            with st.expander("View Model Decision Logic (SHAP)"):
                # Removing the custom icon parameter lets Streamlit use its official native icon
                st.info("**How to read this chart:** Red bars push the client's risk higher, while Blue bars pull their risk lower.")
                
                explainer = shap.TreeExplainer(model)
                shap_explanation = explainer(input_df)
                
                if len(shap_explanation.shape) == 3:
                    shap_values_applicant = shap_explanation[:, :, 1][0]
                else:
                    shap_values_applicant = shap_explanation[0]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_values_applicant, max_display=7, show=False)
                
                st.pyplot(fig)
                plt.clf()
            
        else:
            st.info("Fill out the client's financial profile on the left and click **Predict Default Risk** to generate the prediction result.")
