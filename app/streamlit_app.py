import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import joblib
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebase.firebase_config import save_prediction, get_all_predictions, get_prediction_stats
from firebase.auth import login_user, register_user, reset_password

# ============ Page Configuration ============
st.set_page_config(
    page_title="XAI Risk Decomposition Framework",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Paths ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.json')
DECOMPOSITION_PATH = os.path.join(BASE_DIR, 'data', 'risk_decomposition.json')
IMPORTANCE_PATH = os.path.join(BASE_DIR, 'data', 'feature_importance.csv')
EVAL_PATH = os.path.join(BASE_DIR, 'data', 'evaluation_results.json')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')


# ============ Load Resources ============
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_feature_names():
    with open(FEATURES_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_decomposition():
    with open(DECOMPOSITION_PATH, 'r') as f:
        return json.load(f)

@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)

@st.cache_data
def load_eval_results():
    with open(EVAL_PATH, 'r') as f:
        return json.load(f)


# Load everything
model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()
decomposition = load_decomposition()
importance_df = load_importance()
eval_results = load_eval_results()

RISK_LABELS = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk', 3: 'Critical Risk'}
RISK_COLORS = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c', 'Critical Risk': '#8e44ad'}

# ============ Authentication ============
def show_auth_page():
    st.title("XAI Risk Decomposition Framework")
    st.markdown("### Login or Register to Continue")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.markdown("#### Login")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", key="login_btn"):
            if login_email and login_password:
                result = login_user(login_email, login_password)
                if result["success"]:
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = login_email
                    st.rerun()
                else:
                    st.error(result["error"])
            else:
                st.warning("Please enter email and password")

        if st.button("Forgot Password?", key="forgot_btn"):
            if login_email:
                result = reset_password(login_email)
                if result["success"]:
                    st.success("Password reset email sent")
                else:
                    st.error(result["error"])
            else:
                st.warning("Enter your email first")

    with tab2:
        st.markdown("#### Register")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password (min 6 characters)", type="password", key="reg_password")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")

        if st.button("Register", type="primary", key="reg_btn"):
            if reg_email and reg_password and reg_confirm:
                if reg_password != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    result = register_user(reg_email, reg_password)
                    if result["success"]:
                        st.success("Registration successful. Please login.")
                    else:
                        st.error(result["error"])
            else:
                st.warning("Please fill all fields")


# Check authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    show_auth_page()
    st.stop()

# ============ Custom CSS ============
st.markdown("""
<style>
    .risk-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        text-align: center;
    }
    .low-risk { background: linear-gradient(135deg, #2ecc71, #27ae60); }
    .medium-risk { background: linear-gradient(135deg, #f39c12, #e67e22); }
    .high-risk { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .critical-risk { background: linear-gradient(135deg, #8e44ad, #6c3483); }
    .metric-box {
        background: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ============ Prediction Function ============
def predict_risk(input_data):
    """
    input_data: dict of feature values
    Returns prediction results
    """
    values = [input_data[f] for f in feature_names]
    values_array = np.array(values).reshape(1, -1)
    scaled = scaler.transform(values_array)

    # Repeat for 5 time steps (simulate sequence)
    sequence = np.repeat(scaled, 5, axis=0).reshape(1, 5, len(feature_names))

    prediction = model.predict(sequence, verbose=0)
    pred_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    return {
        'risk_level': pred_class,
        'risk_label': RISK_LABELS[pred_class],
        'confidence': confidence,
        'probabilities': {RISK_LABELS[i]: float(prediction[0][i]) for i in range(4)}
    }


# ============ Sidebar ============
st.sidebar.title("XAI Risk Framework")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Risk Prediction", "XAI Explanations", "Model Performance", "Prediction History", "About"]
)

st.sidebar.markdown(f"**User:** {st.session_state.get('user_email', '')}")
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = ""
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.text(f"Accuracy: {eval_results['accuracy']:.1%}")
st.sidebar.text(f"F1-Score: {eval_results['f1_score']:.2f}")
st.sidebar.text(f"Type: {eval_results['model_type']}")


# ============ PAGE: Dashboard ============
if page == "Dashboard":
    st.title("Organizational Collapse Risk Dashboard")
    st.markdown("Real-time Risk Monitoring and Early Warning System")
    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Model Accuracy", value=f"{eval_results['accuracy']:.1%}")
    with col2:
        st.metric(label="F1-Score", value=f"{eval_results['f1_score']:.2f}")
    with col3:
        st.metric(label="Features Used", value=eval_results['num_features'])
    with col4:
        st.metric(label="Risk Classes", value=eval_results['num_classes'])

    st.markdown("---")

    # Risk Decomposition Overview
    col1, col2 = st.columns(2)

    with col1:
        categories = list(decomposition.keys())
        percentages = [decomposition[c]['percentage'] for c in categories]
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

        fig = px.pie(
            names=categories, values=percentages,
            color_discrete_sequence=colors,
            title='Risk Decomposition Overview'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories, y=percentages,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in percentages],
            textposition='auto'
        ))
        fig.update_layout(
            title='Risk Category Contribution',
            yaxis_title='Contribution (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("### Top Risk Factors")
    top_features = importance_df.head(10)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features['feature'],
        x=top_features['importance'],
        orientation='h',
        marker_color=['#e74c3c' if v > 0 else '#2ecc71' for v in top_features['importance']]
    ))
    fig.update_layout(
        title='Top 10 Features by Importance',
        xaxis_title='Importance (Accuracy Drop)',
        height=400,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show output images from Colab
    st.markdown("### Training Visualizations")
    col1, col2 = st.columns(2)

    eval_img = os.path.join(OUTPUTS_DIR, 'evaluation_results.png')
    risk_img = os.path.join(OUTPUTS_DIR, 'risk_distribution.png')

    if os.path.exists(eval_img):
        with col1:
            st.image(eval_img, caption='Model Evaluation', use_container_width=True)
    if os.path.exists(risk_img):
        with col2:
            st.image(risk_img, caption='Risk Distribution', use_container_width=True)


# ============ PAGE: Risk Prediction ============
elif page == "Risk Prediction":
    st.title("Risk Prediction Engine")
    st.markdown("Enter company financial data to predict risk level")
    st.markdown("---")

    input_method = st.radio("Input Method", ["Manual Input", "Upload CSV"])

    if input_method == "Manual Input":
        
        company_name = st.text_input("Company Name", "My Company")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Liquidity Ratios**")
            current_ratio = st.number_input("Current Ratio", 0.0, 10.0, 1.5, step=0.1)
            cash_flow_ratio = st.number_input("Cash Flow Ratio", -5.0, 10.0, 0.3, step=0.1)
            working_capital = st.number_input("Working Capital / Total Assets", -1.0, 1.0, 0.2, step=0.05)

        with col2:
            st.markdown("**Profitability Ratios**")
            roa = st.number_input("Return on Assets", -1.0, 1.0, 0.05, step=0.01)
            gross_margin = st.number_input("Gross Margin", -1.0, 1.0, 0.3, step=0.05)
            operating_margin = st.number_input("Operating Margin", -1.0, 1.0, 0.1, step=0.05)
            net_income_growth = st.number_input("Net Income Growth", -5.0, 5.0, 0.02, step=0.01)

        with col3:
            st.markdown("**Leverage and Efficiency**")
            debt_to_equity = st.number_input("Debt to Equity", 0.0, 20.0, 1.0, step=0.1)
            equity_to_liabilities = st.number_input("Equity / Total Liabilities", 0.0, 10.0, 0.8, step=0.1)
            interest_coverage = st.number_input("Interest Coverage Ratio", -10.0, 50.0, 3.0, step=0.5)
            ebit_to_assets = st.number_input("EBIT / Total Assets", -1.0, 1.0, 0.08, step=0.01)
            retained_earnings = st.number_input("Retained Earnings / Total Assets", -1.0, 1.0, 0.1, step=0.05)
            revenue_to_assets = st.number_input("Revenue / Total Assets", 0.0, 5.0, 1.2, step=0.1)
            inventory_turnover = st.number_input("Inventory Turnover", 0.0, 50.0, 6.0, step=0.5)
            asset_turnover = st.number_input("Asset Turnover", 0.0, 5.0, 1.0, step=0.1)


        if st.button("Predict Risk Level", type="primary"):
            input_data = {
                'current_ratio': current_ratio,
                'debt_to_equity': debt_to_equity,
                'return_on_assets': roa,
                'working_capital_to_total_assets': working_capital,
                'retained_earnings_to_total_assets': retained_earnings,
                'ebit_to_total_assets': ebit_to_assets,
                'equity_to_total_liabilities': equity_to_liabilities,
                'revenue_to_total_assets': revenue_to_assets,
                'net_income_growth': net_income_growth,
                'cash_flow_ratio': cash_flow_ratio,
                'gross_margin': gross_margin,
                'operating_margin': operating_margin,
                'interest_coverage_ratio': interest_coverage,
                'inventory_turnover': inventory_turnover,
                'asset_turnover': asset_turnover
            }

            result = predict_risk(input_data)

            st.markdown("---")
            st.markdown("### Prediction Result")

            risk_css = result['risk_label'].lower().replace(' ', '-')
            st.markdown(f"""
            <div class="risk-card {risk_css}">
                <h1>{result['risk_label']}</h1>
                <h3>Confidence: {result['confidence']*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            col1, col2 = st.columns(2)

            with col1:
                prob_df = pd.DataFrame({
                    'Risk Level': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                fig = px.bar(
                    prob_df, x='Risk Level', y='Probability',
                    color='Risk Level',
                    color_discrete_map=RISK_COLORS,
                    title='Risk Probability Distribution'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Gauge chart
                risk_score = result['risk_level'] * 25 + result['confidence'] * 25
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': RISK_COLORS[result['risk_label']]},
                        'steps': [
                            {'range': [0, 25], 'color': '#d5f5e3'},
                            {'range': [25, 50], 'color': '#fdebd0'},
                            {'range': [50, 75], 'color': '#fadbd8'},
                            {'range': [75, 100], 'color': '#e8daef'}
                        ]
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
#save to Firebase            
            try:
                save_prediction(company_name, input_data, result)
                st.success(f"Prediction saved to database for {company_name}")
            except Exception as e:
                st.warning(f"Could not save to database: {e}")
                
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Company Financial Data (CSV)", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                results = []
                for idx, row in df.iterrows():
                    try:
                        input_data = {f: row[f] for f in feature_names}
                        result = predict_risk(input_data)
                        results.append({
                            'Index': idx,
                            'Risk Level': result['risk_label'],
                            'Confidence': f"{result['confidence']*100:.1f}%"
                        })
                    except Exception as e:
                        results.append({
                            'Index': idx,
                            'Risk Level': 'Error',
                            'Confidence': str(e)
                        })

                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Summary
                st.markdown("### Batch Summary")
                summary = results_df['Risk Level'].value_counts()
                fig = px.pie(names=summary.index, values=summary.values, title='Batch Risk Distribution')
                st.plotly_chart(fig, use_container_width=True)


# ============ PAGE: XAI Explanations ============
elif page == "XAI Explanations":
    st.title("Explainable AI - Risk Decomposition")
    st.markdown("Understanding why a company is predicted at risk")
    st.markdown("---")

    # Risk Decomposition
    st.markdown("### Risk Factor Decomposition")

    col1, col2 = st.columns(2)

    categories = list(decomposition.keys())
    percentages = [decomposition[c]['percentage'] for c in categories]
    scores = [decomposition[c]['score'] for c in categories]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

    with col1:
        # Radar Chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=percentages + [percentages[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.3)',
            line_color='#e74c3c',
            name='Risk Profile'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(percentages) + 10])),
            title='Risk Radar Chart',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Waterfall Chart
        fig = go.Figure(go.Waterfall(
            name="Risk Decomposition",
            orientation="v",
            measure=["relative"] * len(categories) + ["total"],
            x=categories + ["Total Risk"],
            y=scores + [sum(scores)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#8e44ad"}}
        ))
        fig.update_layout(title="Risk Waterfall Decomposition", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Breakdown
    st.markdown("### Detailed Risk Breakdown")

    for category in categories:
        details = decomposition[category]
        with st.expander(f"{category} - {details['percentage']}%"):
            st.progress(min(details['percentage'] / 100, 1.0))
            st.markdown(f"**Total Score:** {details['score']}")

            if 'features' in details:
                for feat in details['features']:
                    direction = "Increases Risk" if feat['importance'] > 0 else "Decreases Risk"
                    st.text(f"  {feat['feature']}: {feat['importance']:.4f} ({direction})")

    # Feature Importance
    st.markdown("---")
    st.markdown("### Feature Importance (Permutation Method)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance_df['feature'],
        x=importance_df['importance'],
        orientation='h',
        marker_color=['#e74c3c' if v > 0 else '#2ecc71' for v in importance_df['importance']]
    ))
    fig.update_layout(
        title='All Features by Importance',
        xaxis_title='Importance (Accuracy Drop when Feature is Shuffled)',
        height=500,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **How to read this:**
    - Red bars = Shuffling this feature causes accuracy to DROP (important feature)
    - Green bars = Shuffling this feature slightly improves accuracy (less important)
    - Higher bar = More important for prediction
    """)

    # XAI Images from Colab
    st.markdown("---")
    st.markdown("### LIME and SHAP Analysis")

    col1, col2 = st.columns(2)

    xai_img = os.path.join(OUTPUTS_DIR, 'xai_analysis.png')
    lime_img = os.path.join(OUTPUTS_DIR, 'lime_explanation.png')

    if os.path.exists(xai_img):
        with col1:
            st.image(xai_img, caption='Permutation Importance Analysis', use_container_width=True)

    if os.path.exists(lime_img):
        with col2:
            st.image(lime_img, caption='LIME Explanation - Critical Risk Sample', use_container_width=True)


# ============ PAGE: Model Performance ============
elif page == "Model Performance":
    st.title("Model Performance Analysis")
    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{eval_results['accuracy']:.1%}")
    with col2:
        st.metric("F1-Score", f"{eval_results['f1_score']:.2f}")
    with col3:
        st.metric("Training Epochs", eval_results['training_epochs'])
    with col4:
        st.metric("Best Val Accuracy", f"{eval_results['best_val_accuracy']:.1%}")

    st.markdown("---")

    # Model Architecture Info
    st.markdown("### Model Architecture")
    st.markdown("""
    | Component | Details |
    |-----------|---------|
    | **Model Type** | BiLSTM with Multi-Head Attention |
    | **Input Shape** | (5 time steps, 15 features) |
    | **LSTM Layers** | Bidirectional LSTM (64) + Bidirectional LSTM (32) |
    | **Attention** | Multi-Head Attention (2 heads, key_dim=16) |
    | **Dense Layers** | Dense(64) + Dense(32) + Dense(4) |
    | **Output** | 4 classes (Softmax) |
    | **Optimizer** | Adam (learning rate: 0.001) |
    | **Loss** | Sparse Categorical Crossentropy |
    """)

    # Evaluation Images
    st.markdown("### Training and Evaluation Plots")

    eval_img = os.path.join(OUTPUTS_DIR, 'evaluation_results.png')
    if os.path.exists(eval_img):
        st.image(eval_img, caption='Confusion Matrix and Training History', use_container_width=True)

    col1, col2 = st.columns(2)

    eda_img = os.path.join(OUTPUTS_DIR, 'eda_distributions.png')
    corr_img = os.path.join(OUTPUTS_DIR, 'correlation_heatmap.png')

    if os.path.exists(eda_img):
        with col1:
            st.image(eda_img, caption='Feature Distributions', use_container_width=True)
    if os.path.exists(corr_img):
        with col2:
            st.image(corr_img, caption='Feature Correlations', use_container_width=True)

    decomp_img = os.path.join(OUTPUTS_DIR, 'risk_decomposition.png')
    if os.path.exists(decomp_img):
        st.image(decomp_img, caption='Risk Decomposition Analysis', use_container_width=True)


# ============ PAGE: Prediction History ============
elif page == "Prediction History":
    st.title("Prediction History")
    st.markdown("All predictions saved to Firebase")
    st.markdown("---")

    try:
        predictions = get_all_predictions()

        if len(predictions) == 0:
            st.info("No predictions saved yet. Go to Risk Prediction page to make predictions.")
        else:
            # Stats
            stats = get_prediction_stats()
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Predictions", stats['total'])
            with col2:
                st.metric("Low Risk", stats['Low Risk'])
            with col3:
                st.metric("Medium Risk", stats['Medium Risk'])
            with col4:
                st.metric("High Risk", stats['High Risk'])
            with col5:
                st.metric("Critical Risk", stats['Critical Risk'])

            st.markdown("---")

            # History Table
            history_data = []
            for pred in predictions:
                history_data.append({
                    'Company': pred.get('company_name', 'N/A'),
                    'Risk Level': pred.get('risk_label', 'N/A'),
                    'Confidence': f"{pred.get('confidence', 0)*100:.1f}%",
                    'Timestamp': pred.get('timestamp', 'N/A')
                })

            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

            # Chart
            if stats['total'] > 0:
                risk_counts = {k: v for k, v in stats.items() if k != 'total' and v > 0}
                if risk_counts:
                    fig = px.pie(
                        names=list(risk_counts.keys()),
                        values=list(risk_counts.values()),
                        color=list(risk_counts.keys()),
                        color_discrete_map=RISK_COLORS,
                        title='Prediction Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading predictions: {e}")



# ============ PAGE: About ============
elif page == "About":
    st.title("About This Project")
    st.markdown("---")

    st.markdown("""
    ## XAI-Driven Risk Decomposition Framework for Organizational Collapse Prediction

    ### Project Overview
    This project implements a machine learning framework to predict early signs of corporate 
    financial distress using company financial data. The system uses deep learning (BiLSTM with 
    Attention) combined with Explainable AI techniques to not only predict risk levels but also 
    identify the key financial factors contributing to the predicted risk.

    ### Technologies Used
    - **Python** - Core programming language
    - **TensorFlow/Keras** - LSTM model development
    - **Streamlit** - Interactive web dashboard
    - **Firebase** - Cloud database for storing predictions
    - **SHAP/LIME** - Explainable AI techniques
    - **Plotly** - Interactive visualizations

    ### Methodology
    1. **Data Collection**: Financial ratios from company records
    2. **Feature Engineering**: Altman Z-Score calculation, risk labeling
    3. **Model Training**: BiLSTM with Multi-Head Attention mechanism
    4. **Explainability**: Permutation Importance + LIME for risk factor identification
    5. **Risk Decomposition**: Categorizing risk into Liquidity, Leverage, Profitability, 
       Efficiency, and Solvency components

    ### Risk Categories
    | Level | Description |
    |-------|-------------|
    | **Low Risk** | Z-Score > 2.99 - Company is financially healthy |
    | **Medium Risk** | Z-Score 1.81-2.99 - Grey zone, needs monitoring |
    | **High Risk** | Z-Score 1.0-1.81 - Financial distress likely |
    | **Critical Risk** | Z-Score < 1.0 - Collapse imminent |

    ### Key Financial Indicators
    - **Liquidity Risk**: Current Ratio, Cash Flow Ratio, Working Capital
    - **Leverage Risk**: Debt-to-Equity, Interest Coverage, Equity-to-Liabilities
    - **Profitability Risk**: ROA, Gross Margin, Operating Margin
    - **Efficiency Risk**: Asset Turnover, Inventory Turnover
    - **Solvency Risk**: EBIT-to-Assets, Retained Earnings-to-Assets
    """)