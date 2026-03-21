# ============================================
# Digital Twin of Plasma Medicine
# No-Code ML Framework Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Digital Twin of Plasma Medicine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS Styling
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2E75B6;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #EBF3FB;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1F4E79;
        margin: 10px 0;
    }
    .prediction-complete {
        background-color: #C6EFCE;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-strong {
        background-color: #FFEB9C;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-weak {
        background-color: #FFCC99;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-negligible {
        background-color: #FFC7CE;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Models
# ============================================
@st.cache_resource
def load_models():
    with open('best_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('best_regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    return classifier, regressor, scaler, feature_names, label_map

classifier, regressor, scaler, feature_names, label_map = load_models()
reverse_label_map = {v: k for k, v in label_map.items()}

# ============================================
# Header
# ============================================
st.markdown(
    '<div class="main-header">⚡ Digital Twin of Plasma Medicine</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">A No-Code Machine Learning Framework for '
    'Predicting Cold Atmospheric Plasma Efficacy</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/lightning-bolt.png",
    width=80
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home",
     "🔬 Predict Antimicrobial Activity",
     "📊 Model Performance",
     "📚 About"]
)

# ============================================
# PAGE 1: Home
# ============================================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📰 Articles Used", "45", "+12 vs PDF")
    with col2:
        st.metric("🎯 Classification Accuracy", "83.33%", "+0.65% vs PDF")
    with col3:
        st.metric("📈 Regression R²", "0.4824", "146 observations")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 What This Tool Does")
        st.write("""
        This Digital Twin framework uses Machine Learning to predict
        the **antimicrobial activity** of Cold Atmospheric
        Plasma-Activated Liquids (PALs) without any lab experiments.

        Simply enter your plasma parameters and get instant predictions!
        """)

        st.subheader("📊 Models Used")
        model_data = {
            'Task': ['Classification', 'Regression'],
            'Model': ['K-Nearest Neighbors (KNN)', 'Random Forest (RFR)'],
            'Performance': ['83.33% Accuracy', 'R² = 0.4824']
        }
        st.dataframe(pd.DataFrame(model_data), hide_index=True)

    with col2:
        st.subheader("📋 MI Category Scale")
        categories = {
            'Category': ['✅ Complete', '🟡 Strong',
                         '🟠 Weak', '🔴 Negligible'],
            'MI Range': ['≥ 6.0 log', '3.0 – 5.99 log',
                         '1.0 – 2.99 log', '< 1.0 log'],
            'Meaning': ['Full inactivation', 'High inactivation',
                        'Moderate inactivation', 'Minimal inactivation']
        }
        st.dataframe(pd.DataFrame(categories), hide_index=True)

        st.subheader("🔬 Supported Microorganisms")
        microbes = ['E. coli', 'S. aureus', 'P. aeruginosa',
                    'L. monocytogenes', 'S. typhimurium',
                    'E. faecalis', 'C. albicans']
        for m in microbes:
            st.write(f"• {m}")

# ============================================
# PAGE 2: Prediction
# ============================================
elif page == "🔬 Predict Antimicrobial Activity":
    st.header("🔬 Predict Antimicrobial Activity")
    st.write("Enter your plasma parameters below and click Predict!")

    st.divider()

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚡ Plasma Parameters")
        plasma_type = st.selectbox(
            "Plasma Treatment Type",
            ['DBD', 'Plasma Jet', 'Surface DBD',
             'Volume DBD', 'Gliding Arc']
        )
        gas_type = st.selectbox(
            "Gas Type",
            ['Air', 'Argon', 'Nitrogen', 'Oxygen',
             'Helium', 'Helium+Oxygen', 'Argon+Oxygen']
        )
        treatment_time = st.slider(
            "Plasma Treatment Time (seconds)",
            min_value=0, max_value=3600,
            value=600, step=60
        )

    with col2:
        st.subheader("💧 Liquid Parameters")
        liquid_type = st.selectbox(
            "Plasma Activated Liquid",
            ['Deionized Water', 'Saline', 'PBS',
             'Ringer Solution', 'Tap Water']
        )
        treatment_volume = st.slider(
            "Treatment Volume (mL)",
            min_value=0, max_value=100,
            value=5, step=1
        )
        contact_time = st.slider(
            "Contact Time (minutes)",
            min_value=0, max_value=60,
            value=10, step=1
        )

    with col3:
        st.subheader("🦠 Microorganism Parameters")
        microbial_strain = st.selectbox(
            "Microbial Strain",
            ['E.coli', 'S.aureus', 'P.aeruginosa',
             'L.monocytogenes', 'S.typhimurium',
             'E.faecalis', 'C.albicans']
        )
        initial_load = st.slider(
            "Initial Microbial Load (log CFU/mL)",
            min_value=4, max_value=9,
            value=7, step=1
        )
        incubation_temp = st.slider(
            "Incubation Temperature (°C)",
            min_value=4, max_value=37,
            value=25, step=1
        )
        post_storage = st.slider(
            "Post Storage Time (minutes)",
            min_value=0, max_value=43200,
            value=0, step=60
        )

    st.divider()

    # Predict Button
    if st.button("🚀 PREDICT ANTIMICROBIAL ACTIVITY",
                  type="primary", use_container_width=True):

        # Build input dataframe
        input_data = pd.DataFrame([{
            'Plasma_Treatment_Type': plasma_type,
            'Gas_Type': gas_type,
            'Plasma_Treatment_Time_s': treatment_time,
            'Plasma_Activated_Liquid': liquid_type,
            'Treatment_Volume_mL': treatment_volume,
            'Microbial_Strain': microbial_strain,
            'Initial_Microbial_Load_log': initial_load,
            'Contact_Time_min': contact_time,
            'Incubation_Temperature_C': incubation_temp,
            'Post_Storage_Time_min': post_storage
        }])

        # One-hot encode
        cat_cols = ['Plasma_Treatment_Type', 'Gas_Type',
                    'Plasma_Activated_Liquid', 'Microbial_Strain']
        num_cols = ['Plasma_Treatment_Time_s', 'Treatment_Volume_mL',
                    'Initial_Microbial_Load_log', 'Contact_Time_min',
                    'Incubation_Temperature_C', 'Post_Storage_Time_min']

        input_encoded = pd.get_dummies(input_data, columns=cat_cols)

        # Align with training features
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]

        # Scale numeric features
        input_encoded[num_cols] = scaler.transform(
            input_encoded[num_cols]
        )

        # Predict
        clf_pred = classifier.predict(input_encoded)[0]
        reg_pred = regressor.predict(input_encoded)[0]
        category = reverse_label_map[clf_pred]

        # Show results
        st.success("✅ Prediction Complete!")
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Classification Result")
            if category == 'Complete':
                st.markdown(
                    f'<div class="prediction-complete">'
                    f'✅ {category}<br>Full Inactivation</div>',
                    unsafe_allow_html=True
                )
            elif category == 'Strong':
                st.markdown(
                    f'<div class="prediction-strong">'
                    f'🟡 {category}<br>High Inactivation</div>',
                    unsafe_allow_html=True
                )
            elif category == 'Weak':
                st.markdown(
                    f'<div class="prediction-weak">'
                    f'🟠 {category}<br>Moderate Inactivation</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-negligible">'
                    f'🔴 {category}<br>Minimal Inactivation</div>',
                    unsafe_allow_html=True
                )

        with col2:
            st.subheader("📈 Regression Result")
            st.metric(
                "Predicted MI Value",
                f"{reg_pred:.2f} log",
                f"Category: {category}"
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reg_pred,
                title={'text': "MI Value (log)"},
                gauge={
                    'axis': {'range': [0, 9]},
                    'bar': {'color': "#1F4E79"},
                    'steps': [
                        {'range': [0, 1], 'color': '#FFC7CE'},
                        {'range': [1, 3], 'color': '#FFCC99'},
                        {'range': [3, 6], 'color': '#FFEB9C'},
                        {'range': [6, 9], 'color': '#C6EFCE'}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("📋 Input Summary")
            summary = {
                'Parameter': [
                    'Plasma Type', 'Gas Type',
                    'Treatment Time', 'Liquid Type',
                    'Volume', 'Microbial Strain',
                    'Initial Load', 'Contact Time',
                    'Temperature', 'Storage Time'
                ],
                'Value': [
                    plasma_type, gas_type,
                    f"{treatment_time}s", liquid_type,
                    f"{treatment_volume}mL", microbial_strain,
                    f"{initial_load} log", f"{contact_time} min",
                    f"{incubation_temp}°C", f"{post_storage} min"
                ]
            }
            st.dataframe(
                pd.DataFrame(summary),
                hide_index=True,
                use_container_width=True
            )

# ============================================
# PAGE 3: Model Performance
# ============================================
elif page == "📊 Model Performance":
    st.header("📊 Model Performance Dashboard")

    tab1, tab2 = st.tabs(
        ["🤖 Classification", "📈 Regression"]
    )

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "83.33%", "+0.65% vs PDF")
        col2.metric("F1 Score", "0.8269")
        col3.metric("Recall", "0.8375")
        col4.metric("AUC", "0.9292")

        st.subheader("All Classifiers Comparison")
        clf_data = pd.DataFrame({
            'Model': ['KNN','LDA','DTC','LR','ETC','SVM',
                      'GPC','RFC','ABC','BC','GBC','XGBC',
                      'BNB','GNB'],
            'Test Accuracy': [0.8333, 0.8000, 0.8000,
                               0.7667, 0.7667, 0.7667,
                               0.7333, 0.7333, 0.7333,
                               0.7333, 0.7333, 0.7333,
                               0.6667, 0.4333]
        })
        fig = px.bar(
            clf_data, x='Model', y='Test Accuracy',
            color='Test Accuracy',
            color_continuous_scale='Blues',
            title='Classification Models - Test Accuracy'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", "0.4824")
        col2.metric("MAE", "0.9766")
        col3.metric("RMSE", "1.2976")

        st.subheader("All Regressors Comparison")
        reg_data = pd.DataFrame({
            'Model': ['ABR','RFR','XGBR','BaR','GBR',
                      'ETR','LSVR','MLPR','RR','BR',
                      'KNR','ENR','LLars','LASSO'],
            'R2 Score': [0.5818, 0.5815, 0.5558,
                          0.5493, 0.5316, 0.5007,
                          0.4919, 0.4673, 0.4557,
                          0.4461, 0.3623, -0.0602,
                          -0.0756, -0.0756]
        })
        fig2 = px.bar(
            reg_data, x='Model', y='R2 Score',
            color='R2 Score',
            color_continuous_scale='RdYlGn',
            title='Regression Models - R² Score'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 4: About
# ============================================
elif page == "📚 About":
    st.header("📚 About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Project Title")
        st.info(
            "Digital Twin of Plasma Medicine: A No-Code Machine "
            "Learning Framework for Predicting Cold Atmospheric "
            "Plasma Efficacy"
        )

        st.subheader("📊 Dataset")
        st.write("""
        - **45 published articles** (2008-2025)
        - **146 observations** after cleaning
        - **10 input features**
        - **4 MI categories:** Negligible, Weak, Strong, Complete
        """)

        st.subheader("🔬 Based On")
        st.write("""
        Özdemir et al. (2023) - *Machine learning to predict the
        antimicrobial activity of cold atmospheric
        plasma-activated liquids*
        Machine Learning: Science and Technology, 4, 015030
        """)

    with col2:
        st.subheader("🚀 Novel Contributions")
        st.write("""
        1. ✅ **First No-Code Dashboard** for PAL prediction
        2. ✅ **Extended Dataset** (45 vs 33 articles)
        3. ✅ **Better Accuracy** (83.33% vs 82.68%)
        4. ✅ **New MI Category** (Negligible added)
        5. ✅ **Interactive Digital Twin** interface
        6. ✅ **Real-time predictions** for researchers
        """)

        st.subheader("🛠️ Technology Stack")
        tech = {
            'Tool': ['Python', 'Scikit-learn', 'Streamlit',
                     'Pandas', 'Plotly', 'GitHub'],
            'Purpose': ['Programming', 'ML Models', 'Dashboard',
                        'Data Processing', 'Visualizations',
                        'Deployment']
        }
        st.dataframe(pd.DataFrame(tech), hide_index=True)

# Footer
st.divider()
st.markdown(
    "<center>Digital Twin of Plasma Medicine | "
    "Built with Streamlit ⚡</center>",
    unsafe_allow_html=True
)
```

### Step 3
- Click **"Commit new file"** ✅

---

## ✅ Step 10.4 — Upload PKL Files to GitHub

- Go to your GitHub repository
- Click **"Add file" → "Upload files"**
- Upload all 5 files:
```
best_classifier.pkl
best_regressor.pkl
scaler.pkl
feature_names.pkl
label_map.pkl
