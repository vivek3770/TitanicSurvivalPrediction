# 🚢 Titanic Survival Predictor

**A Machine Learning project with Jupyter Notebook + Streamlit, powered by the famous Titanic Dataset (Kaggle).**

Predict Titanic passenger survival in seconds! Train the model in a notebook, then interact with it through a sleek Streamlit web app.

---

## ✨ Features

- **Notebook Training:** Explore the dataset, preprocess features, and train a Random Forest model in `model.ipynb`.
- **Live Prediction (Streamlit):** Enter age, ticket class, family info, and get instant survival predictions with probabilities.
- **Modern UI:** Dark mode with cards, sidebar navigation, and custom design for clarity.
- **Feature Insights:** Visual bar chart showing which factors matter most.
- **Profile Card:** Review every input and prediction in a neat result card.
- **Clean, Reproducible Code:** Modular Python with pipelines for scaling, encoding, and imputing missing data.

---

## 🚀 Getting Started

1. **Clone this repository**
    ```bash
    git clone https://github.com/vivek3770/Codsoft.git
    cd Codsoft/TITANICSURVIVALPREDICTION
    ```

2. **Install requirements**
    ```bash
    pip install streamlit pandas scikit-learn matplotlib numpy joblib
    ```

3. **Train the model in Jupyter Notebook**
    - Open `model.ipynb` in Jupyter/VS Code.
    - Run all cells to preprocess the dataset and train the Random Forest model.
    - This will save the trained model as `titanic_model.pkl`.

4. **Launch the Streamlit app**
    ```bash
    streamlit run app.py
    ```

5. **Open your browser** to the shown local address and interact with the app!

---

## 🧑💻 How it works

- **Notebook (`model.ipynb`):**
  - Loads and preprocesses the Titanic dataset.
  - Builds a pipeline with imputers, scalers, encoders, and a Random Forest classifier.
  - Trains the model and saves it as `titanic_model.pkl`.

- **App (`app.py`):**
  - Loads the trained model.
  - Provides a modern Streamlit UI with sidebar navigation.
  - Allows users to input passenger details and see survival predictions instantly.
  - Displays feature importance with a bar chart.
  - Shows a profile card with all passenger inputs.

---

## 📂 Project Structure

TITANICSURVIVALPREDICTION/
│── app.py               # Streamlit app with UI
│── model.ipynb          # Jupyter Notebook for training
│── TitanicDataset.csv   # Dataset (Kaggle Titanic dataset)
│── titanic_model.pkl   # Saved trained model
│── README.md            # Project documentation


---

## 📊 Features Used

- **Pclass** – Ticket class (1st, 2nd, 3rd)  
- **Sex** – Passenger gender  
- **Age** – Passenger age in years  
- **SibSp** – Number of siblings/spouses aboard  
- **Parch** – Number of parents/children aboard  
- **Fare** – Ticket fare (£)  
- **Embarked** – Boarding port (S = Southampton, C = Cherbourg, Q = Queenstown)  

---

## 🙌 Credits

- **Dataset:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- **Created by:** Vivek