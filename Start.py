import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")


st.markdown("""
    <style>
    /* Background and main app container */
    .stApp {
        background: #1e293b;  /* dark blue-gray */
        color: #cbd5e1;       /* light blue-gray text */
        font-family: 'Inter', sans-serif;
    }
    /* Card style for better contrast */
    .card {
        background: #334155;  /* slightly lighter dark card */
        border-radius: 16px;
        padding: 2em 2em 1.5em 2em;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        color: #e0e7ff;
    }
    /* Header styling */
    .app-header {
        padding: 32px 0 14px 0;
        background: #475569;
        color: #f1f5f9;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 28px;
    }
    /* Section titles */
    .section-title {
        font-size: 1.6em;
        font-weight: 700;
        color: #f8fafc;
        margin: 12px 0 19px 0;
    }
    /* Prediction result colors */
    .result-success {
        color: #22c55e;  /* green */
        font-weight: 700;
        font-size: 1.2em;
        margin-bottom: 8px;
    }
    .result-fail {
        color: #ef4444; /* red */
        font-weight: 700;
        font-size: 1.2em;
        margin-bottom: 8px;
    }
    /* Button style overrides */
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



@st.cache_resource
def fit_model():
    df = pd.read_csv("Titanic-Dataset.csv")
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].nunique() < 10]
    num_cols = [c for c in X.columns if X[c].dtype in ['int64','float64'] and c not in cat_cols]
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42))
    ])
    model.fit(X, y)
    importances = model.named_steps['classifier'].feature_importances_
    return model, list(X.columns), cat_cols, num_cols, importances

model, X_columns, categorical_cols, numerical_cols, feature_importance = fit_model()

feature_explanation = {
    "Pclass": "Ticket class (1=1st, 3=3rd)", "Sex": "Passenger gender",
    "Age": "Passenger age in years", "SibSp": "Number of siblings/spouses aboard",
    "Parch": "Number of parents/children aboard", "Fare": "Ticket fare (£)",
    "Embarked": "Boarding port (S: Southampton, C: Cherbourg, Q: Queenstown)"
}

# --- Sidebar menu ---
st.sidebar.image("https://cdn.wallpapersafari.com/34/81/uV7PmB.jpg", use_container_width=True)
menu = st.sidebar.radio("Menu", ["Prediction", "Feature Importance", "About"], key='menu', label_visibility='visible')

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

    # --- Prediction ---
    if st.button("🔮 Predict Survival", use_container_width=True):
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[X_columns]
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction]

        # --- Animated Result Card ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-title">Prediction Result</span>', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown('<div class="result-success">🎉 <b>Likely to Survive!</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fail">😞 <b>Not Likely to Survive.</b></div>', unsafe_allow_html=True)
        st.markdown(f'<p class="big-number">{proba:.1%} chance</p>', unsafe_allow_html=True)
        st.progress(int(proba * 100))
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Profile Card ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-title">Passenger Profile</span>', unsafe_allow_html=True)
        profile_icon = "👦" if sex=="male" else "👧"
        st.markdown(f"<div class='profile-icon'>{profile_icon}</div>", unsafe_allow_html=True)
        st.table(input_df.T.rename(columns={0:'Value'}))
        st.markdown('</div>', unsafe_allow_html=True)

       
       

        

elif menu == "Feature Importance":
    st.markdown('<div class="app-header"><h2>💡 Feature Importance</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fi_sorted = sorted(
        [(feat, val) for feat, val in zip(X_columns, feature_importance) if feat.lower() != "passengerid"],
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

elif menu == "About":
    st.markdown('<div class="app-header"><h2>ℹ️ About</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b>Model</b>: Random Forest, scikit-learn<br>
    <b>Dataset</b>: Titanic Kaggle Dataset<br>
    <b>Engine</b>: Streamlit + modern data science UX<br>
    <br>
    <b>Enter a passenger profile to see their survival prediction,<br>
    plus what features matter most to the model—instantly!</b>
    <br><br>
    <i><br>Created by Vivek.</i>
    </div>
    """, unsafe_allow_html=True)
