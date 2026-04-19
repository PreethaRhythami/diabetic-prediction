
Live Demo: https://preetha-diabetes-prediction.streamlit.appDiabetic Prediction Web App

A machine learning web application that predicts the likelihood of diabetes in a patient based on clinical input parameters. Built with Python, Scikit-learn, and Streamlit.

Overview

This project uses the Pima Indians Diabetes Dataset to train a Random Forest classifier that predicts whether a patient is diabetic based on health metrics such as glucose level, BMI, age, and more. The app includes user authentication, real-time prediction, and a downloadable PDF health report.

Features

1.User registration and login (SQLite)

2.Real-time diabetes risk prediction

3.Model performance visualizations (ROC curve, confusion matrix)

4.Downloadable PDF health report for each prediction

5.Clean and interactive Streamlit UI

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Framework | Scikit-learn |
| Web Framework | Streamlit |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Report Generation | FPDF / ReportLab |

---

## Model Performance

| Metric | Score |
|---|---|
| Algorithm | Random Forest Classifier |
| Accuracy | ~85% |
| Dataset | Pima Indians Diabetes Dataset |

> See `conf_matrix.png` and `roc_diabetes.jpeg` in the repo for detailed model evaluation visuals.

---

## Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/PreethaRhythami/diabetic-prediction.git
cd diabetic-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

Project Structure

diabetic-prediction/
│
├── app.py                  # Main Streamlit application
├── database.py             # User authentication logic (SQLite)
├── diabetes.csv            # Dataset
├── diabetes_modell.pkl     # Trained Random Forest model
├── feature_names.pkl       # Saved feature names for prediction
├── conf_matrix.png         # Confusion matrix visualization
├── roc_diabetes.jpeg       # ROC curve visualization
└── README.md

Dataset

- Source:[Pima Indians Diabetes Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features:Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: Outcome (0 = Non-diabetic, 1 = Diabetic)
