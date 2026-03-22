import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Digital Twin of Plasma Medicine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ==============================
       LIGHT MODE
    ============================== */
    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: #F0F4F8;
        }
        .main .block-container {
            background-color: #F0F4F8;
        }
        .main-header {
            color: #1F4E79 !important;
        }
        .sub-header {
            color: #5B8DB8 !important;
        }
        h1, h2, h3, h4, p, label {
            color: #1F4E79 !important;
        }
        .sidebar-ref-box {
            background-color: #FFFFFF;
            border: 1px solid #2E75B6;
            border-radius: 10px;
            padding: 15px;
            color: #1F4E79 !important;
        }
        .sidebar-ref-box p,
        .sidebar-ref-box a {
            color: #1F4E79 !important;
        }
    }

    /* ==============================
       DARK MODE
    ============================== */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0E1117;
        }
        .main .block-container {
            background-color: #0E1117;
        }
        .main-header {
            color: #63B3ED !important;
        }
        .sub-header {
            color: #90CDF4 !important;
        }
        h1, h2, h3, h4, p, label {
            color: #E2E8F0 !important;
        }
        .sidebar-ref-box {
            background-color: #1A202C;
            border: 1px solid #63B3ED;
            border-radius: 10px;
            padding: 15px;
            color: #E2E8F0 !important;
        }
        .sidebar-ref-box p,
        .sidebar-ref-box a {
            color: #90CDF4 !important;
        }
        div[data-testid="metric-container"] {
            background-color: #1A202C !important;
            border-left: 4px solid #63B3ED !important;
        }
    }

    /* ==============================
       SHARED STYLES (both modes)
    ============================== */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        padding: 25px 0 5px 0;
        letter-spacing: 1px;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 10px;
        font-style: italic;
    }
    .prediction-complete {
        background: linear-gradient(135deg, #1a7a4a, #2ECC71);
        color: white !important;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(46,204,113,0.4);
        margin: 10px 0;
    }
    .prediction-strong {
        background: linear-gradient(135deg, #b8860b, #F4D03F);
        color: white !important;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(244,208,63,0.4);
        margin: 10px 0;
    }
    .prediction-weak {
        background: linear-gradient(135deg, #c0392b, #E74C3C);
        color: white !important;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(231,76,60,0.4);
        margin: 10px 0;
    }
    .prediction-negligible {
        background: linear-gradient(135deg, #6c3483, #9B59B6);
        color: white !important;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(155,89,182,0.4);
        margin: 10px 0;
    }
    div[data-testid="metric-container"] {
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2E75B6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 15px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2E75B6, #1F4E79);
        box-shadow: 0 4px 15px rgba(31,78,121,0.4);
        transform: translateY(-2px);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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

REFERENCE = "Ozdemir et al. (2023)"

# ============================================
# Header
# ============================================
st.markdown(
    '<div class="main-header">Digital Twin of Plasma Medicine</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">No-Code Machine Learning Framework for '
    'Predicting Cold Atmospheric Plasma Efficacy</div>',
    unsafe_allow_html=True
)
st.divider()

# ============================================
# Sidebar
# ============================================
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "",
    ["Home", "Predict", "Model Performance", "About"]
)
st.sidebar.divider()
st.sidebar.markdown("### Reference Study")
st.sidebar.markdown("""
<div class="sidebar-ref-box">
    <p><strong>This work extends:</strong></p>
    <p><strong>Ozdemir et al. (2023)</strong></p>
    <p>Machine Learning: Science and Technology, 4, 015030</p>
    <p>DOI: 10.1088/2632-2153/acc1c0</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE 1: Home
# ============================================
if page == "Home":

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Research Articles", "45",
            f"Extended from {REFERENCE}"
        )
    with col2:
        st.metric(
            "Observations", "146",
            f"Dataset: {REFERENCE}"
        )
    with col3:
        st.metric(
            "Classification Accuracy", "83.33%",
            "KNN Model"
        )
    with col4:
        st.metric(
            "Regression R2", "0.48",
            "RFR Model"
        )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("About This Tool")
        st.write("""
        This Digital Twin framework uses Machine Learning to predict
        the antimicrobial activity of Cold Atmospheric
        Plasma-Activated Liquids (PALs) without requiring
        laboratory experiments.

        Simply enter your plasma treatment parameters and receive
        instant predictions of microbial inactivation efficacy.

        This work extends the foundational study by
        Ozdemir et al. (2023) by introducing an interactive
        no-code dashboard accessible to all researchers.
        """)

        st.subheader("ML Models")
        model_data = {
            'Task': ['Classification', 'Regression'],
            'Model': [
                'K-Nearest Neighbors (KNN)',
                'Random Forest Regressor (RFR)'
            ],
            'Performance': ['83.33% Accuracy', 'R2 = 0.48'],
            'Reference': [REFERENCE, REFERENCE]
        }
        st.dataframe(
            pd.DataFrame(model_data),
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.subheader("Microbial Inactivation Scale")
        categories = {
            'Category': ['Complete', 'Strong',
                         'Weak', 'Negligible'],
            'MI Range': [
                'MI >= 6.0 log',
                'MI 3.0 - 5.99 log',
                'MI 1.0 - 2.99 log',
                'MI < 1.0 log'
            ],
            'Interpretation': [
                'Full bacterial inactivation',
                'High bacterial inactivation',
                'Moderate bacterial inactivation',
                'Minimal bacterial inactivation'
            ]
        }
        st.dataframe(
            pd.DataFrame(categories),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Supported Microorganisms")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("- E. coli")
            st.write("- S. aureus")
            st.write("- P. aeruginosa")
            st.write("- L. monocytogenes")
        with col_b:
            st.write("- S. typhimurium")
            st.write("- E. faecalis")
            st.write("- C. albicans")

# ============================================
# PAGE 2: Predict
# ============================================
elif page == "Predict":
    st.header("Predict Antimicrobial Activity")
    st.write(
        "Fill in the parameters below and click "
        "Predict to get your results."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Plasma Parameters")
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
        st.subheader("Liquid Parameters")
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
        st.subheader("Microorganism Parameters")
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
            "Incubation Temperature (C)",
            min_value=4, max_value=37,
            value=25, step=1
        )
        post_storage = st.slider(
            "Post Storage Time (minutes)",
            min_value=0, max_value=43200,
            value=0, step=60
        )

    st.divider()

    if st.button(
        "PREDICT ANTIMICROBIAL ACTIVITY",
        type="primary",
        use_container_width=True
    ):
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

        cat_cols = ['Plasma_Treatment_Type', 'Gas_Type',
                    'Plasma_Activated_Liquid', 'Microbial_Strain']
        num_cols = ['Plasma_Treatment_Time_s', 'Treatment_Volume_mL',
                    'Initial_Microbial_Load_log', 'Contact_Time_min',
                    'Incubation_Temperature_C', 'Post_Storage_Time_min']

        input_encoded = pd.get_dummies(input_data, columns=cat_cols)
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]
        input_encoded[num_cols] = scaler.transform(
            input_encoded[num_cols]
        )

        clf_pred = classifier.predict(input_encoded)[0]
        reg_pred = regressor.predict(input_encoded)[0]
        category = reverse_label_map[clf_pred]

        st.success("Prediction Complete!")
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Classification Result")
            if category == 'Complete':
                st.markdown(
                    '<div class="prediction-complete">'
                    'COMPLETE<br>'
                    '<span style="font-size:1rem">'
                    'Full Bacterial Inactivation'
                    '</span></div>',
                    unsafe_allow_html=True
                )
            elif category == 'Strong':
                st.markdown(
                    '<div class="prediction-strong">'
                    'STRONG<br>'
                    '<span style="font-size:1rem">'
                    'High Bacterial Inactivation'
                    '</span></div>',
                    unsafe_allow_html=True
                )
            elif category == 'Weak':
                st.markdown(
                    '<div class="prediction-weak">'
                    'WEAK<br>'
                    '<span style="font-size:1rem">'
                    'Moderate Bacterial Inactivation'
                    '</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-negligible">'
                    'NEGLIGIBLE<br>'
                    '<span style="font-size:1rem">'
                    'Minimal Bacterial Inactivation'
                    '</span></div>',
                    unsafe_allow_html=True
                )

        with col2:
            st.subheader("Predicted MI Value")
            st.metric(
                "Microbial Inactivation",
                f"{reg_pred:.2f} log reduction"
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(reg_pred),
                title={'text': "MI Value (log reduction)"},
                gauge={
                    'axis': {'range': [0, 9]},
                    'bar': {'color': "#1F4E79"},
                    'steps': [
                        {'range': [0, 1], 'color': '#EBD5F5'},
                        {'range': [1, 3], 'color': '#FADBD8'},
                        {'range': [3, 6], 'color': '#FCF3CF'},
                        {'range': [6, 9], 'color': '#D5F5E3'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': reg_pred
                    }
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("Input Summary")
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
                    f"{treatment_time} s", liquid_type,
                    f"{treatment_volume} mL", microbial_strain,
                    f"{initial_load} log", f"{contact_time} min",
                    f"{incubation_temp} C", f"{post_storage} min"
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
elif page == "Model Performance":
    st.header("Model Performance")
    st.caption(
        f"Results based on dataset extended from {REFERENCE}"
    )
    st.divider()

    tab1, tab2 = st.tabs(["Classification", "Regression"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "83.33%", "KNN Model")
        col2.metric("F1 Score", "0.8269", "KNN Model")
        col3.metric("Recall", "0.8375", "KNN Model")
        col4.metric("AUC", "0.9292", "KNN Model")

        st.divider()
        st.subheader("All Classifiers - Test Accuracy")
        clf_data = pd.DataFrame({
            'Model': ['KNN', 'LDA', 'DTC', 'LR', 'ETC',
                      'SVM', 'GPC', 'RFC', 'ABC', 'BC',
                      'GBC', 'XGBC', 'BNB', 'GNB'],
            'Test Accuracy': [
                0.8333, 0.8000, 0.8000, 0.7667, 0.7667,
                0.7667, 0.7333, 0.7333, 0.7333, 0.7333,
                0.7333, 0.7333, 0.6667, 0.4333
            ]
        })
        fig = px.bar(
            clf_data, x='Model', y='Test Accuracy',
            color='Test Accuracy',
            color_continuous_scale='Blues',
            title='Classification Models Performance'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("R2 Score", "0.4824", "RFR Model")
        col2.metric("MAE", "0.9766", "RFR Model")
        col3.metric("RMSE", "1.2976", "RFR Model")

        st.divider()
        st.subheader("All Regressors - R2 Score")
        reg_data = pd.DataFrame({
            'Model': ['ABR', 'RFR', 'XGBR', 'BaR', 'GBR',
                      'ETR', 'LSVR', 'MLPR', 'RR', 'BR',
                      'KNR', 'ENR', 'LLars', 'LASSO'],
            'R2 Score': [
                0.5818, 0.5815, 0.5558, 0.5493, 0.5316,
                0.5007, 0.4919, 0.4673, 0.4557, 0.4461,
                0.3623, -0.0602, -0.0756, -0.0756
            ]
        })
        fig2 = px.bar(
            reg_data, x='Model', y='R2 Score',
            color='R2 Score',
            color_continuous_scale='RdYlGn',
            title='Regression Models Performance'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 4: About
# ============================================
elif page == "About":
    st.header("About This Project")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Title")
        st.info(
            "Digital Twin of Plasma Medicine: A No-Code Machine "
            "Learning Framework for Predicting Cold Atmospheric "
            "Plasma Efficacy"
        )

        st.subheader("Dataset Summary")
        st.write("""
        - 45 published research articles (2008 - 2025)
        - 146 experimental observations
        - 10 input features
        - 4 output categories: Negligible, Weak, Strong, Complete
        """)

        st.subheader("Reference Study")
        st.write("""
        Ozdemir et al. (2023)
        Machine learning to predict the antimicrobial activity
        of cold atmospheric plasma-activated liquids.
        Machine Learning: Science and Technology, 4, 015030.
        DOI: 10.1088/2632-2153/acc1c0
        """)

    with col2:
        st.subheader("Key Contributions")
        st.write("""
        1. First interactive no-code dashboard for PAL prediction
        2. Extended literature dataset (2008 - 2025)
        3. New Negligible MI category introduced
        4. Real-time prediction without lab experiments
        5. Interactive Digital Twin interface
        6. Accessible to non-coding researchers
        """)

        st.subheader("Technology Stack")
        tech = {
            'Tool': ['Python', 'Scikit-learn', 'Streamlit',
                     'Pandas', 'Plotly', 'GitHub'],
            'Purpose': [
                'Core Programming',
                'Machine Learning Models',
                'Web Dashboard',
                'Data Processing',
                'Interactive Visualizations',
                'Deployment'
            ]
        }
        st.dataframe(
            pd.DataFrame(tech),
            hide_index=True,
            use_container_width=True
        )

# ============================================
# Footer
# ============================================
st.divider()
st.markdown(
    "<center style='font-size: 0.9rem;'>"
    "Digital Twin of Plasma Medicine | "
    "No-Code ML Framework | 2025 | "
    f"Extending {REFERENCE}"
    "</center>",
    unsafe_allow_html=True
)
