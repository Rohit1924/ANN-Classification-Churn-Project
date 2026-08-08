import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 14px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model("model.h5")


Model = load_model()


# =========================================================
# LOAD ENCODERS AND SCALER
# =========================================================

@st.cache_resource
def load_preprocessors():

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("oneHOt_encoder.pkl", "rb") as file:
        oneHOt_encoder = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return label_encoder_gender, oneHOt_encoder, scaler


label_encoder_gender, oneHOt_encoder, scaler = load_preprocessors()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered system to predict whether a bank customer is likely to churn'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown("### 👤 Customer Information")

col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    Geography = st.selectbox(
        "🌍 Geography",
        oneHOt_encoder.categories_[0]
    )

    gender = st.selectbox(
        "⚧ Gender",
        label_encoder_gender.classes_
    )

    Age = st.slider(
        "🎂 Age",
        18,
        92,
        35
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=0,
        max_value=900,
        value=650
    )

    balance = st.number_input(
        "💰 Balance",
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


# =========================================================
# COLUMN 3
# =========================================================

with col3:

    tenure = st.slider(
        "📅 Tenure",
        0,
        10,
        3
    )

    num_of_product = st.slider(
        "📦 Number of Products",
        1,
        4,
        2
    )

    has_cr_card = st.selectbox(
        "💳 Has Credit Card",
        [0, 1]
    )

    is_active_number = st.selectbox(
        "⚡ Is Active Member",
        [0, 1]
    )


# =========================================================
# GENDER ENCODING
# =========================================================

gender_encoded = label_encoder_gender.transform(
    [gender]
)[0]


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [Age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_product],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_number],
        "EstimatedSalary": [estimated_salary]
    }
)


# =========================================================
# GEOGRAPHY ENCODING
# =========================================================

geo_encoder = oneHOt_encoder.transform(
    [[Geography]]
).toarray()


geo_encoder_df = pd.DataFrame(
    geo_encoder,
    columns=oneHOt_encoder.get_feature_names_out(
        ["Geography"]
    )
)


# =========================================================
# COMBINE INPUT DATA
# =========================================================

input_data = pd.concat(
    [
        input_data,
        geo_encoder_df
    ],
    axis=1
)


# =========================================================
# SCALE INPUT
# =========================================================

input_data_scaled = scaler.transform(
    input_data
)


# =========================================================
# MODEL PREDICTION
# =========================================================

prediction = Model.predict(
    input_data_scaled,
    verbose=0
)

prediction_proba = float(
    prediction[0][0]
)


# =========================================================
# PREDICTION RESULT
# =========================================================

st.markdown("### 🔮 Prediction Result")

st.metric(
    label="Churn Probability",
    value=f"{prediction_proba:.2%}"
)


# =========================================================
# CHURN STATUS
# =========================================================

if prediction_proba > 0.5:

    st.error(
        "⚠️ The customer is likely to churn."
    )

else:

    st.success(
        "✅ The customer is not likely to churn."
    )


# =========================================================
# PROGRESS BAR
# =========================================================

st.progress(
    prediction_proba,
    text=f"Churn Risk: {prediction_proba:.2%}"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🤖 Powered by TensorFlow • Scikit-learn • Pandas • Streamlit
        <br>
        Customer Churn Prediction — Deep Learning Project
    </div>
    """,
    unsafe_allow_html=True
)