import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: #1e293b;
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
    }
    .card {
        background: #334155;
        border-radius: 16px;
        padding: 2em 2em 1.5em 2em;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        color: #e0e7ff;
    }
    .app-header {
        padding: 32px 0 14px 0;
        background: #475569;
        color: #f1f5f9;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 28px;
    }
    .section-title {
        font-size: 1.6em;
        font-weight: 700;
        color: #f8fafc;
        margin: 12px 0 19px 0;
    }
    .result-success {
        color: #22c55e;
        font-weight: 700;
        font-size: 1.2em;
        margin-bottom: 8px;
    }
    .result-fail {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.2em;
        margin-bottom: 8px;
    }
    .stButton button {
        border-radius: 10px !important;
        font-size: 1.16em !important;
        background-color: #2563eb !important;
        color: white !important;
        padding: 10px 0 !important;
    }
    .stButton button:hover {
        background-color: #1d4ed8 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
model, X_columns = joblib.load("titanic_model.pkl")
feature_explanation = {
    "Pclass": "Ticket class (1=1st, 3=3rd)", "Sex": "Passenger gender",
    "Age": "Passenger age in years", "SibSp": "Number of siblings/spouses aboard",
    "Parch": "Number of parents/children aboard", "Fare": "Ticket fare (£)",
    "Embarked": "Boarding port (S: Southampton, C: Cherbourg, Q: Queenstown)"
}

# --- Sidebar ---
st.sidebar.image("https://cdn.wallpapersafari.com/34/81/uV7PmB.jpg", use_container_width=True)
menu = st.sidebar.radio("Menu", ["Prediction", "Feature Importance", "About"], key='menu')

# --- Prediction ---
if menu == "Prediction":
    st.markdown('<div class="app-header"><h1>🚢 Titanic Survival Predictor</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-title">Passenger Details</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Ticket Class", [1,2,3], help=feature_explanation["Pclass"])
        sex = st.radio("Sex", ["male", "female"], help=feature_explanation["Sex"])
        age = st.slider("Age", 0, 80, 29, help=feature_explanation["Age"])
        sibsp = st.number_input("Siblings/Spouses", 0, 8, 0, help=feature_explanation["SibSp"])
    with col2:
        parch = st.number_input("Parents/Children", 0, 6, 0, help=feature_explanation["Parch"])
        fare = st.number_input("Fare (£)", 0.0, 600.0, 32.2, help=feature_explanation["Fare"])
        embarked = st.radio("Embarked", ["S", "C", "Q"], help=feature_explanation["Embarked"]) if "Embarked" in X_columns else None

    input_dict = {'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare}
    if 'Embarked' in X_columns: input_dict['Embarked'] = embarked
    input_df = pd.DataFrame([input_dict])
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict Survival", use_container_width=True):
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[X_columns]
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-title">Prediction Result</span>', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown('<div class="result-success">🎉 <b>Likely to Survive!</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fail">😞 <b>Not Likely to Survive.</b></div>', unsafe_allow_html=True)
        st.markdown(f'<p class="big-number">{proba:.1%} chance</p>', unsafe_allow_html=True)
        st.progress(int(proba * 100))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-title">Passenger Profile</span>', unsafe_allow_html=True)
        profile_icon = "👦" if sex=="male" else "👧"
        st.markdown(f"<div class='profile-icon'>{profile_icon}</div>", unsafe_allow_html=True)
        st.table(input_df.T.rename(columns={0:'Value'}))
        st.markdown('</div>', unsafe_allow_html=True)

# --- Feature Importance ---
elif menu == "Feature Importance":
    st.markdown('<div class="app-header"><h2>💡 Feature Importance</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    importances = model.named_steps['classifier'].feature_importances_
    fi_sorted = sorted(
        [(feat, val) for feat, val in zip(X_columns, importances) if feat.lower() != "passengerid"],
        key=lambda x: -x[1]
    )
    fi_features, fi_vals = zip(*fi_sorted[:7])
    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.barh(fi_features, fi_vals, color="#2665a7")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.grid(axis='x', alpha=0.22)
    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-title">Feature Explanations</span>', unsafe_allow_html=True)
    for k, v in feature_explanation.items():
        if k in X_columns: st.markdown(f"- **{k}**: {v}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- About ---
elif menu == "About":
    st.markdown('<div class="app-header"><h2>ℹ️ About</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Model</b>: Random Forest, scikit-learn<br>
    <b>Dataset</b>: Titanic Kaggle Dataset<br>
    <b>Engine</b>: Streamlit + modern data science UX<br><br>
    <b>Enter a passenger profile to see their survival prediction,<br>
    plus what features matter most to the model—instantly!</b><br><br>
    <i>Created by Vivek.</i>
    </div>
    """, unsafe_allow_html=True)
