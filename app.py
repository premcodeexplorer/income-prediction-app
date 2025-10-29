"""
ADULT INCOME PREDICTION - STREAMLIT FRONTEND (FIXED)
============================================
Interactive web application for income prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Income Predictor 💰",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED for better visibility
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Better button styling */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    
    /* Fix text visibility */
    .stMarkdown, .stText, p, span, div {
        color: #1f1f1f !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1f1f1f !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Input labels */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #1f1f1f !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* AGGRESSIVE FIX FOR DROPDOWN - Target all possible selectors */
    
    /* Main selectbox container */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    /* All divs with data-baseweb attribute */
    [data-baseweb] {
        background-color: #ffffff !important;
    }
    
    /* Specific baseweb components */
    [data-baseweb="select"],
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="list"],
    [data-baseweb="list-item"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    /* All list elements */
    ul[role="listbox"],
    ul[role="menu"],
    li[role="option"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    /* All divs with role attributes */
    div[role="listbox"],
    div[role="option"],
    div[role="menu"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    /* Hover states */
    li[role="option"]:hover,
    div[role="option"]:hover,
    [data-baseweb="list-item"]:hover {
        background-color: #e3f2fd !important;
        color: #000000 !important;
    }
    
    /* Target by class if exists */
    .css-10trblm,
    .css-16idsys,
    .css-1n76uvr {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    /* Dropdown text */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] span,
    [data-baseweb="menu"] span,
    [data-baseweb="list-item"] span {
        color: #1f1f1f !important;
    }
    
    /* Number input - Multiple selectors for better targeting */
    .stNumberInput > div > div > input {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    .stNumberInput input {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    input[type="number"] {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    /* Text input fields */
    .stTextInput > div > div > input {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    .stTextInput input {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    input[type="text"] {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    /* All input fields */
    input {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #1f1f1f !important;
        font-size: 14px !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
        font-size: 24px !important;
    }
    
    /* Tab text */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 600;
    }
    
    /* Dataframe */
    .dataframe {
        color: #1f1f1f !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load all saved models and preprocessing objects"""
    try:
        # Load from models folder
        with open('models/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        with open('models/preprocessing.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load model comparison
        model_comparison = pd.read_csv('models/model_comparison.csv')
        
        return best_model, preprocessing, scaler, model_comparison
    
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.info("Please ensure all .pkl files are in the 'models/' folder")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None, None

# Load models
best_model, preprocessing, scaler, model_comparison = load_models()

if best_model is None:
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📊 Model Information")
    
    if model_comparison is not None:
        best_model_name = model_comparison.iloc[0]['Model']
        best_accuracy = model_comparison.iloc[0]['Accuracy']
        best_f1 = model_comparison.iloc[0]['F1-Score']
        
        st.success(f"**Best Model:** {best_model_name}")
        st.metric("Accuracy", f"{best_accuracy:.2%}")
        st.metric("F1-Score", f"{best_f1:.2%}")
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This app predicts income levels using machine learning.
    
    **Features:**
    - 6 ML algorithms compared
    - Hyperparameter tuned
    - Real-time predictions
    """)
    
    st.markdown("---")
    st.header("📚 Dataset")
    st.write("""
    **UCI Adult Income**
    - 48,842 samples
    - 14 features
    - Binary classification
    """)

# ============================================================================
# MAIN HEADER
# ============================================================================
st.title("💰 Adult Income Prediction System")
st.markdown("### *Predict whether annual income exceeds $50K*")
st.markdown("---")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Performance", "📈 About Project"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.header("Make a Prediction")
    st.write("Fill in the information below to get an income prediction:")
    st.markdown("")
    
    # Create 3 columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Personal Information")
        age = st.slider("Age", 17, 90, 35, help="Person's age in years")
        sex = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", [
            "White", "Black", "Asian-Pac-Islander", 
            "Amer-Indian-Eskimo", "Other"
        ])
        marital_status = st.selectbox("Marital Status", [
            "Never-married", "Married-civ-spouse", "Divorced",
            "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"
        ])
        relationship = st.selectbox("Relationship", [
            "Husband", "Not-in-family", "Own-child",
            "Unmarried", "Wife", "Other-relative"
        ])
    
    with col2:
        st.markdown("#### 💼 Work Information")
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Local-gov",
            "State-gov", "Self-emp-inc", "Federal-gov",
            "Without-pay", "Never-worked"
        ])
        occupation = st.selectbox("Occupation", [
            "Prof-specialty", "Exec-managerial", "Craft-repair",
            "Sales", "Adm-clerical", "Other-service",
            "Machine-op-inspct", "Tech-support", "Transport-moving",
            "Handlers-cleaners", "Farming-fishing", "Protective-serv",
            "Priv-house-serv", "Armed-Forces"
        ])
        hours_per_week = st.slider("Work Hours per Week", 1, 99, 40)
        fnlwgt = st.number_input("Final Weight (Census)", 
                                  min_value=10000, 
                                  max_value=1500000, 
                                  value=200000,
                                  help="Census demographic weight - typically leave as default")
    
    with col3:
        st.markdown("#### 🎓 Education & Finance")
        education = st.selectbox("Education Level", [
            "Bachelors", "HS-grad", "Some-college", "Masters",
            "Assoc-voc", "11th", "Assoc-acdm", "10th",
            "7th-8th", "Prof-school", "9th", "12th",
            "Doctorate", "5th-6th", "1st-4th", "Preschool"
        ])
        
        capital_gain = st.number_input("Capital Gain ($)", 0, 100000, 0,
                                       help="Money from investment sources")
        capital_loss = st.number_input("Capital Loss ($)", 0, 5000, 0,
                                       help="Money lost from investments")
        
        native_country = st.selectbox("Native Country", [
            "United-States", "Mexico", "Philippines", "Germany",
            "Puerto-Rico", "Canada", "India", "Cuba", "England",
            "China", "South", "Jamaica", "Italy", "Poland",
            "Vietnam", "Japan", "Taiwan", "Iran", "Greece",
            "Other"
        ])
    
    # Predict button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("🔮 PREDICT INCOME LEVEL", use_container_width=True)
    
    if predict_button:
        
        with st.spinner("Making prediction..."):
            # Create input dictionary
            input_data = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': native_country
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Feature Engineering (same as training)
            input_df['age_group'] = pd.cut(input_df['age'], 
                                            bins=[0, 25, 35, 45, 55, 100],
                                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            
            # Education num mapping
            education_num_map = {
                'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
                '9th': 5, '10th': 6, '11th': 7, '12th': 8,
                'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11,
                'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
                'Prof-school': 15, 'Doctorate': 16
            }
            education_num = education_num_map.get(education, 9)
            
            input_df['work_intensity'] = education_num * input_df['hours-per-week']
            input_df['net_capital'] = input_df['capital-gain'] - input_df['capital-loss']
            input_df['has_capital'] = (input_df['net_capital'] != 0).astype(int)
            
            # Encode categorical features
            label_encoders = preprocessing['label_encoders']
            for col in label_encoders.keys():
                if col in input_df.columns:
                    le = label_encoders[col]
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except:
                        # Handle unseen categories
                        input_df[col] = 0
            
            # Get feature names and reorder
            feature_names = preprocessing['feature_names']
            
            # Add missing columns
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Select and reorder features
            input_df = input_df[feature_names]
            
            # Make prediction
            prediction = best_model.predict(input_df)[0]
            prediction_proba = best_model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        st.markdown("")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                    <div style='background-color: #d4edda; padding: 30px; border-radius: 15px; 
                                border-left: 8px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h2 style='color: #155724; margin: 0; font-size: 28px;'>✅ Income > $50K</h2>
                        <p style='color: #155724; margin: 15px 0 0 0; font-size: 18px;'>
                            Predicted with <b style='font-size: 22px;'>{prediction_proba[1]*100:.1f}%</b> confidence
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #f8d7da; padding: 30px; border-radius: 15px; 
                                border-left: 8px solid #dc3545; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h2 style='color: #721c24; margin: 0; font-size: 28px;'>❌ Income ≤ $50K</h2>
                        <p style='color: #721c24; margin: 15px 0 0 0; font-size: 18px;'>
                            Predicted with <b style='font-size: 22px;'>{prediction_proba[0]*100:.1f}%</b> confidence
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba[1] * 100,
                title={'text': "Probability of Income >$50K", 'font': {'size': 18, 'color': '#1f1f1f'}},
                number={'font': {'size': 32, 'color': '#1f1f1f'}, 'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#1f1f1f'},
                    'bar': {'color': "#4CAF50" if prediction == 1 else "#f44336"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffebee"},
                        {'range': [50, 100], 'color': "#e8f5e9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#1f1f1f'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence breakdown
        st.markdown("---")
        st.markdown("#### 📈 Confidence Breakdown")
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            st.metric("Income ≤ $50K", f"{prediction_proba[0]:.2%}", 
                     delta=None, delta_color="normal")
        
        with conf_col2:
            st.metric("Income > $50K", f"{prediction_proba[1]:.2%}",
                     delta=None, delta_color="normal")

# ============================================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================================
with tab2:
    st.header("📊 Model Performance Comparison")
    st.write("Comparison of all 6 machine learning models trained on this dataset")
    st.markdown("")
    
    if model_comparison is not None:
        st.subheader("📋 Complete Results Table")
        st.dataframe(model_comparison.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                    use_container_width=True, height=250)
        
        st.markdown("")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(model_comparison, x='Model', y='Accuracy',
                        title='Accuracy Comparison Across Models',
                        color='Accuracy',
                        color_continuous_scale='Blues',
                        text='Accuracy')
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': '#1f1f1f'},
                title_font_size=16,
                xaxis_title="Model",
                yaxis_title="Accuracy",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(model_comparison, x='Model', y='F1-Score',
                        title='F1-Score Comparison Across Models',
                        color='F1-Score',
                        color_continuous_scale='Greens',
                        text='F1-Score')
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': '#1f1f1f'},
                title_font_size=16,
                xaxis_title="Model",
                yaxis_title="F1-Score",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        st.markdown("---")
        st.subheader(f"🏆 Best Performing Model: {best_model_name}")
        st.markdown("")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Accuracy", f"{model_comparison.iloc[0]['Accuracy']:.2%}")
        with metrics_col2:
            st.metric("Precision", f"{model_comparison.iloc[0]['Precision']:.2%}")
        with metrics_col3:
            st.metric("Recall", f"{model_comparison.iloc[0]['Recall']:.2%}")
        with metrics_col4:
            st.metric("F1-Score", f"{model_comparison.iloc[0]['F1-Score']:.2%}")
        
        st.info(f"✨ **{best_model_name}** achieved the best overall performance with {model_comparison.iloc[0]['Accuracy']:.2%} accuracy!")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab3:
    st.header("📈 About This Project")
    st.markdown("")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This is a comprehensive Machine Learning classification system built to predict 
    whether an individual's annual income exceeds $50,000 based on census data.
    
    ### 📊 Dataset Information
    - **Source**: UCI Adult Income Dataset (Census Income Database)
    - **Total Samples**: 48,842 individuals
    - **Features**: 14 demographic and employment attributes
    - **Target Variable**: Binary classification (>$50K vs ≤$50K annual income)
    - **Class Distribution**: Imbalanced (~76% ≤$50K, ~24% >$50K)
    
    ### 🤖 Machine Learning Models
    
    Six different algorithms were implemented and compared:
    
    1. **Logistic Regression** - Linear baseline classifier
    2. **Decision Tree** - Non-linear tree-based model
    3. **Random Forest** - Ensemble of decision trees
    4. **Support Vector Machine (SVM)** - RBF kernel classifier
    5. **XGBoost** ⭐ - Gradient boosting (Best performer)
    6. **LightGBM** - Microsoft's fast gradient boosting
    
    ### 🔧 Technical Features
    
    ✅ **Comprehensive EDA** - 15+ visualizations and statistical analyses  
    ✅ **Feature Engineering** - Created 4 derived features (age_group, work_intensity, net_capital, has_capital)  
    ✅ **Hyperparameter Tuning** - GridSearchCV and RandomizedSearchCV optimization  
    ✅ **Model Comparison** - Multi-metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
    ✅ **Interactive GUI** - Real-time predictions with confidence scores  
    ✅ **Production Ready** - Saved models with pickle for deployment  
    
    ### 📈 Performance Results
    
    - **Best Model**: XGBoost Classifier
    - **Accuracy**: 87.29%
    - **F1-Score**: 0.7147
    - **Precision**: High (minimizes false positives)
    - **Training Time**: ~5 minutes on standard hardware
    
    ### 💡 Real-World Applications
    
    - **Economic Policy**: Resource allocation and poverty alleviation programs
    - **Financial Services**: Credit risk assessment and loan approval
    - **Research**: Socioeconomic studies and income inequality analysis
    - **HR & Recruitment**: Salary benchmarking and compensation planning
    - **Government**: Targeted assistance and welfare program eligibility
    
    ### 🛠️ Technology Stack
    
    - **Languages**: Python 3.x
    - **ML Libraries**: Scikit-learn, XGBoost, LightGBM
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Web Framework**: Streamlit
    - **Model Persistence**: Pickle
    
    ---
    
    ### 👨‍💻 Project Details
    
    **Author**: [Your Name]  
    **Date**: October 2024  
    **Purpose**: ML Mini Project / Portfolio Demonstration  
    **Code**: Available on GitHub [link]
    """)
    
    st.success("""
    ✅ **Project Highlights**
    - Complete ML pipeline from EDA to deployment
    - 6 algorithms with hyperparameter optimization
    - Professional web interface with real-time predictions
    - Production-ready code with proper documentation
    - Comprehensive evaluation and model comparison
    """)
    
    st.warning("""
    ⚠️ **Disclaimer**: This model is for educational and demonstration purposes. 
    Predictions should not be used for actual employment, financial, or legal decisions 
    without proper validation and compliance with applicable regulations.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666; padding: 20px;'>
        <p style='font-size: 16px; margin: 5px;'>💰 <b>Adult Income Prediction System</b></p>
        <p style='font-size: 14px; margin: 5px;'>Built with Streamlit, Scikit-learn, XGBoost & LightGBM</p>
        <p style='font-size: 12px; margin: 5px;'>Made with ❤️ for ML Mini Project</p>
    </div>
""", unsafe_allow_html=True)