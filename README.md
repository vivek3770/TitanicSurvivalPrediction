# 🚢 Titanic Survival Predictor

**An interactive web app with Machine Learning + Streamlit, powered by the famous Titanic Dataset (Kaggle).**

Predict Titanic passenger survival in seconds! This app lets anyone enter passenger details and see instant predictions, probabilities, and feature importance in a sleek modern UI.

## ✨ Features

- **Live Prediction:** Enter age, ticket class, family info, and get instant survival predictions with probabilities.
- **Modern UI:** Dark mode with cards, sidebar navigation, and custom design for clarity.
- **Feature Insights:** Visual bar chart showing which factors matter most.
- **Profile Card:** Review every input and prediction in a neat result card.
- **Clean, Reproducible Code:** Modular Python with data science best practices.

## 🚀 Getting Started

1. **Clone this repository**
    ```
    git clone https://github.com/vivek3770/Codsoft.git
    cd Codsoft
    ```


2. **Install requirements**
    ```
    pip install streamlit pandas scikit-learn matplotlib numpy
    ```

3. **Launch the app**
    ```
    streamlit run Start.py
    ```

Open your browser to the shown local address and interact with the app!

## 🧑💻 How it works

- **Model:** Random Forest classifier (scikit-learn), with pipelines for scaling, encoding, and imputing missing data.
- **Features Used:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
- **UI:** Streamlit (custom theming, responsive layout, charting).

