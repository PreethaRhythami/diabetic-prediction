import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
try:
    import shap
except ImportError:
    shap = None
    print("Warning: 'shap' is not installed. Some analytics features will be unavailable.")
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: 'matplotlib' is not installed. Plots will be unavailable.")
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: 'seaborn' is not installed. Some plots will be unavailable.")
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sqlite3
from database import init_db, add_user, login_user

init_db() 

if not hasattr(np, 'bool'):
    np.bool = bool

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

if "register" not in st.session_state:
    st.session_state.register = False

if not st.session_state.auth:
    encoded_bg = base64.b64encode(open("bg1.jpg", "rb").read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_bg}") !important;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .center-wrapper {{
            position: relative;
            top: 60px;
            margin: 0 auto;
            width: 90%;
            max-width: 500px;
        }}
        .login-title {{
            text-align: center;
            color: black;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: 0.5px;
            text-shadow: 0px 3px 6px rgba(0, 0, 0, 0.5);
        }}
        .login-subtitle {{
            text-align: center;
            color: black;
            font-size: 1.2rem;
            margin-bottom: 1.2rem;
        }}
        .login-form {{
            background: rgba(0, 0, 0, 0);
            padding: 2rem;
            border-radius: 15px;
        }}
        .stTextInput input {{
            background-color: rgba(255,255,255,0.2);
            color: black;
            border-radius: 10px;
        }}
        .stButton>button {{
            background-color: #00c6ff;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            width: 100%;
            padding: 10px;
            margin-top: 1rem;
        }}
        .switch-link {{
            text-align: center;
            margin-top: 1rem;
        }}
        .switch-link a {{
            color: #ffffff;
            font-weight: bold;
            cursor: pointer;
        }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='center-wrapper'>
            <div class='login-title'>🩺 Diabetes Prediction</div>
            <div class='login-subtitle'>{}</div>
            <div class='login-form'>
    """.format("Login Form" if not st.session_state.register else "Register Form"), unsafe_allow_html=True)

    if not st.session_state.register:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        login_clicked = st.button("Login")

        if login_clicked:
            if login_user(user, pwd):
                st.session_state.auth = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

        if st.button("Go to Register"):
            st.session_state.register = True
            st.rerun()

    else:
        new_user = st.text_input("New Username")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if new_pwd != confirm_pwd:
                st.error("Passwords do not match")
            else:
                success = add_user(new_user, new_pwd)
                if success:
                    st.success("Registration successful. You can now log in.")
                    st.session_state.register = False
                    st.rerun()
                else:
                    st.error("Username already exists.")
        if st.button("Back to Login"):
            st.session_state.register = False
            st.rerun()

    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

model = joblib.load('diabetes_modell.pkl')
expected_features = joblib.load('rf_features.pkl')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()
if len(y_test.shape) > 1:
    y_test = y_test.iloc[:, 0]

# Sidebar navigation
st.sidebar.title("🔀 Navigation")
app_mode = st.sidebar.radio("Go to", ["🩺 Prediction App", "📈 Advanced Analytics"])

# Metrics and confusion matrix
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_mat).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("conf_matrix.png")
plt.close()

st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
st.sidebar.write(f"**Precision:** {precision:.2f}")
st.sidebar.write(f"**Recall:** {recall:.2f}")
st.sidebar.write(f"**F1-Score:** {f1:.2f}")
st.sidebar.image("conf_matrix.png", caption="Confusion Matrix")

if app_mode == "🩺 Prediction App":
    st.title("🩺 Diabetes Prediction App")
    st.write("Fill in the details below and click Predict.")

    patient_name = st.text_input("Name")
    patient_id = st.text_input("ID")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0)
            glucose = st.number_input("Glucose", min_value=0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0)
            skin_thickness = st.number_input("Skin Thickness", min_value=0)
            insulin = st.number_input("Insulin", min_value=0)
        with col2:
            bmi = st.number_input("BMI", min_value=0.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
            age = st.number_input("Age", min_value=0)
            bmi_cat = st.selectbox("BMI Category", ["Obesity 1", "Obesity 2", "Obesity 3", "Overweight", "Underweight"])
            gluc_cat = st.selectbox("Glucose Category", ["Low", "Normal", "Overweight", "Secret"])
            insulin_norm = st.radio("Is Insulin in Normal Range", ["Yes", "No"])
        submit = st.form_submit_button("🔍 Predict")

    if submit:
        # 🧮 Automatically categorize BMI
        if bmi >= 30 and bmi < 35:
            bmi_cat = "Obesity 1"
        elif bmi >= 35 and bmi < 40:
            bmi_cat = "Obesity 2"
        elif bmi >= 40:
            bmi_cat = "Obesity 3"
        elif bmi >= 25:
            bmi_cat = "Overweight"
        else:
            bmi_cat = "Underweight"

        # 🧪 Automatically categorize Glucose
        if glucose < 70:
            gluc_cat = "Low"
        elif 70 <= glucose <= 140:
            gluc_cat = "Normal"
        elif 141 <= glucose <= 180:
            gluc_cat = "Overweight"
        else:
            gluc_cat = "Secret"

        # 💉 Automatically determine if insulin is normal (normal range: 16 to 166)
        if 16 <= insulin <= 166:
            insulin_norm = "Yes"
        else:
            insulin_norm = "No"

        # 👁️ Show user the auto-detected categories
        st.markdown(f"🧮 **BMI Category:** `{bmi_cat}`")
        st.markdown(f"🧪 **Glucose Category:** `{gluc_cat}`")
        st.markdown(f"💉 **Insulin Range Status:** `{insulin_norm}`")
        bmi_enc = {"Obesity 1": [1, 0, 0, 0, 0], "Obesity 2": [0, 1, 0, 0, 0], "Obesity 3": [0, 0, 1, 0, 0], "Overweight": [0, 0, 0, 1, 0], "Underweight": [0, 0, 0, 0, 1]}[bmi_cat]
        glucose_enc = {"Low": [1, 0, 0, 0], "Normal": [0, 1, 0, 0], "Overweight": [0, 0, 1, 0], "Secret": [0, 0, 0, 1]}[gluc_cat]
        insulin_enc = [1] if insulin_norm == "Yes" else [0]

        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age, *bmi_enc, *insulin_enc, *glucose_enc]])
        input_df = pd.DataFrame(input_data, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age',
            'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight', 'NewBMI_Underweight',
            'NewInsulinScore_Normal', 'NewGlucose_Low', 'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'])
        
        # Ensure input features match expected features
        missing_cols = set(expected_features) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[expected_features]  # reorder columns to match expected order

        prediction = model.predict(input_df)[0]  # Use the properly formatted input_df

        st.markdown("### Prediction Result")

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        if result == "Diabetic":
            st.error("🚨 Prediction: The person is Diabetic")
            if glucose > 180 or bmi > 30:
                risk_level = "Severe"
            elif glucose > 140 or bmi > 25:
                risk_level = "Moderate"
            else:
                risk_level = "Mild"
            care_notes = f"Risk Level: {risk_level}. Maintain healthy lifestyle, monitor glucose, follow balanced diet, exercise and take medication."
            result_color = (255, 0, 0)
        else:
            st.success("✅ Prediction: The person is Not Diabetic")
            care_notes = "Risk Level: Low. Continue healthy lifestyle, regular check-ups, and balanced diet."
            result_color = (0, 153, 0)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(0, 102, 204)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 15, "Diabetes Prediction Report", ln=True, align='C', fill=True)
        pdf.ln(20)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, f"Name: {patient_name}", ln=True)
        pdf.cell(100, 10, f"ID: {patient_id}", ln=True)
        pdf.cell(100, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 10, (
            "Diabetes is a chronic health condition that affects how your body turns food into energy. "
            "Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. "
            "When blood sugar goes up, it signals your pancreas to release insulin. If you have diabetes, your body "
            "either doesn't make enough insulin or can't use it as well as it should. Over time, this can cause serious "
            "health problems, such as heart disease, vision loss, and kidney disease.\n\n"
            "Early detection through testing is essential to managing or even reversing the progression of diabetes. "
            "This report aims to provide a predictive analysis based on key health indicators entered above."
        ))

        try:
            pdf.image("risk.jpg", x=30, y=180, w=150)
            pdf.ln(90)
        except:
            pass

        pdf.image("bg.jpg", x=10, y=8, w=30)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Parameter Details", ln=True, align='C')
        pdf.set_fill_color(255, 204, 255)
        pdf.cell(90, 10, "Parameter", 1, 0, 'C', True)
        pdf.cell(90, 10, "Value", 1, 1, 'C', True)
        pdf.set_font("Arial", '', 12)

        fields = [("Pregnancies", pregnancies), ("Glucose", glucose), ("Blood Pressure", blood_pressure),
                  ("Skin Thickness", skin_thickness), ("Insulin", insulin), ("BMI", bmi),
                  ("Diabetes Pedigree Function", dpf), ("Age", age)]

        for label, value in fields:
            pdf.set_fill_color(255, 255, 255)
            pdf.cell(90, 10, str(label), 1)
            pdf.cell(90, 10, str(value), 1, 1)

        pdf.set_fill_color(255, 255, 153)
        pdf.set_text_color(*result_color)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(180, 10, f"Prediction Result: {result}", ln=True, fill=True)

        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 10, f"Health Advice:\n{care_notes}")

        pdf_file = f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_file)

        if submit and os.path.exists(pdf_file):
            with st.spinner("🔄 Preparing download..."):
                time.sleep(1.5)
                with open(pdf_file, "rb") as f:
                    st.download_button("⬇️ Download Report", f.read(), file_name=pdf_file, mime="application/pdf")
                st.success("✅ Report generated successfully! Your download is ready.")
            

elif app_mode == "📈 Advanced Analytics":
    st.title("📈 Advanced Analytics")

    st.subheader("📌 Feature Importance - Global")
    try:
        if shap is None:
            st.warning("SHAP package is not installed. Please install it using 'pip install shap'")
        else:
            with st.spinner("Computing feature importance..."):
                # Get feature importance from the random forest model directly
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                # Create a bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.bar(range(len(feature_importance)), feature_importance['importance'])
                plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
                plt.title('Feature Importance from Random Forest')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Display top 10 features in a table
                st.write("Top 10 Most Important Features:")
                st.table(feature_importance.head(10).style.format({'importance': '{:.4f}'}))
                
    except Exception as e:
        st.error(f"Error in feature importance computation: {str(e)}")
        st.info("Try installing or upgrading SHAP: pip install shap --upgrade")

    # 🧪 Raw Data Display
    st.subheader("🔢 Raw Data Sample")
    st.dataframe(X_test.head())

    # 📊 Cross-Validation Performance
    st.subheader("📊 Cross-Validation Performance")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

# Subheader for section
    st.subheader("🧪 Cross-Validation Scores (Random Forest)")

# Define Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Define Stratified K-Fold Cross Validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run cross-validation on test data (you can change to X_train, y_train if preferred)
    scores = cross_val_score(rf_model, X_test, y_test, cv=cv, scoring='accuracy')

# Display raw scores and summary stats
    st.write("Cross-validation Accuracy Scores:", scores)
    st.write(f"Mean Accuracy: {scores.mean():.4f}")
    st.write(f"Standard Deviation: {scores.std():.4f}")

# Convert scores to DataFrame for plotting
    scores_df = pd.DataFrame(scores, columns=["Accuracy"])

# Boxplot inside the tab
    st.markdown("### 📦 Cross-Validation Score Distribution (Boxplot)")
    fig_box, ax_box = plt.subplots()
    sns.boxplot(data=scores_df, palette="pastel", ax=ax_box)
    ax_box.set_title("K-Fold Cross-Validation Performance (n=5)")
    st.pyplot(fig_box)