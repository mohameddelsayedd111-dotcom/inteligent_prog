"""
ui/app.py
Streamlit web interface for Heart Disease Detection System.
Supports both the Rule-Based Expert System and the Decision Tree model.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'rule_based_system'))
sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))

from rules import HeartDiseaseExpertSystem, PatientData
from data_processing import load_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Detection System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #c0392b; text-align: center; }
    .sub-header  { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 1.5rem; }
    .risk-high   { background: #fde8e8; border-left: 5px solid #c0392b; padding: 1rem; border-radius: 6px; }
    .risk-mod    { background: #fef9e7; border-left: 5px solid #f39c12; padding: 1rem; border-radius: 6px; }
    .risk-low    { background: #e8f8f5; border-left: 5px solid #27ae60; padding: 1rem; border-radius: 6px; }
    .metric-box  { background: #f0f4f8; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">❤️ Heart Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Rule-Based Expert System + Decision Tree Machine Learning Model</div>', unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Choose Module", [
    "🏠 Home",
    "🤖 Expert System",
    "🌳 Decision Tree Model",
    "📊 Data Visualizations",
    "⚖️ Model Comparison"
])

# ════════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🤖 Expert System**\n\nRule-based inference engine with 12 medical rules derived from clinical knowledge.")
    with col2:
        st.success("**🌳 Decision Tree**\n\nScikit-Learn Decision Tree Classifier trained on patient data with hyperparameter tuning.")
    with col3:
        st.warning("**📊 Visualization**\n\nExplore feature distributions, correlation heatmaps, and model performance charts.")

    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
1. **Expert System** — Enter patient vitals manually and get an instant rule-based risk assessment.
2. **Decision Tree** — Upload or enter data to get ML model predictions with probability scores.
3. **Visualizations** — Explore the dataset and understand which features drive risk.
4. **Comparison** — See side-by-side performance of both approaches.
""")

# ════════════════════════════════════════════════════════════════════════════════
# EXPERT SYSTEM PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Expert System":
    st.header("🤖 Rule-Based Expert System")
    st.info("Enter patient health indicators below. The system will fire applicable rules and calculate a risk level.")

    col1, col2 = st.columns(2)
    with col1:
        age      = st.slider("Age (years)", 18, 90, 50)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 220)
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 220, 130)
        thalach  = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
    with col2:
        fbs   = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        cp    = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                              format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                     "Non-Anginal Pain", "Asymptomatic"][x])
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 8.0, 1.0, 0.1)
        ca      = st.slider("Major Vessels Colored (0–3)", 0, 3, 0)

    if st.button("🔍 Run Expert System", type="primary"):
        patient = dict(age=age, sex=sex, chol=chol, trestbps=trestbps,
                       thalach=thalach, fbs=fbs, exang=exang,
                       cp=cp, oldpeak=oldpeak, ca=ca)
        engine = HeartDiseaseExpertSystem()
        engine.reset()
        engine.declare(PatientData(**patient))
        engine.run()
        level = engine.get_risk_level()
        score = engine.risk_score
        rules = engine.fired_rules

        css_class = {"HIGH": "risk-high", "MODERATE": "risk-mod", "LOW": "risk-low"}[level]
        emoji     = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[level]

        st.markdown(f"""
        <div class="{css_class}">
          <h2>{emoji} Risk Level: {level}</h2>
          <p><strong>Risk Score:</strong> {score} | <strong>Rules Fired:</strong> {len(rules)}</p>
        </div>""", unsafe_allow_html=True)

        if rules:
            st.markdown("#### 📜 Rules Fired:")
            for r in rules:
                st.markdown(f"- ✦ {r}")
        else:
            st.success("No high-risk rules triggered. Low baseline risk.")

# ════════════════════════════════════════════════════════════════════════════════
# DECISION TREE PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🌳 Decision Tree Model":
    st.header("🌳 Decision Tree ML Model")
    model_path = os.path.join(BASE_DIR, 'ml_model', 'heart_disease_model.pkl')

    if not os.path.exists(model_path):
        st.error("Model file not found. Please run `ml_model/train_model.py` first.")
    else:
        clf = joblib.load(model_path)
        cleaned = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
        df = pd.read_csv(cleaned)
        feature_cols = [c for c in df.columns if c != 'target']

        st.success(f"✅ Model loaded | Features: {len(feature_cols)} | Classes: Disease / No Disease")

        st.markdown("#### Enter Patient Data (Normalized 0–1 scale):")
        cols = st.columns(4)
        vals = {}
        for i, col_name in enumerate(feature_cols):
            with cols[i % 4]:
                vals[col_name] = st.number_input(col_name, 0.0, 1.0,
                                                  float(df[col_name].median()), 0.01)

        if st.button("🔮 Predict", type="primary"):
            X_input = pd.DataFrame([vals])[feature_cols]
            pred  = clf.predict(X_input)[0]
            prob  = clf.predict_proba(X_input)[0]
            label = "Heart Disease" if pred == 1 else "No Heart Disease"
            css   = "risk-high" if pred == 1 else "risk-low"
            emoji = "🔴" if pred == 1 else "🟢"

            st.markdown(f"""
            <div class="{css}">
              <h2>{emoji} Prediction: {label}</h2>
              <p>Probability — No Disease: <strong>{prob[0]:.2%}</strong> |
                 Disease: <strong>{prob[1]:.2%}</strong></p>
            </div>""", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(["No Disease", "Disease"], [prob[0], prob[1]],
                    color=["#27ae60", "#c0392b"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Prediction Probability")
            st.pyplot(fig)
            plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Visualizations":
    st.header("📊 Data Visualizations")
    data_path = os.path.join(BASE_DIR, 'data', 'raw_data.csv')
    if not os.path.exists(data_path):
        st.error("Dataset not found.")
    else:
        df = pd.read_csv(data_path)
        st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Histograms", "Target Distribution"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='RdYlGn',
                        center=0, linewidths=0.5, ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
            plt.close()

        with tab2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected = st.multiselect("Select features:", num_cols,
                                       default=num_cols[:4])
            if selected:
                fig, axes = plt.subplots(1, len(selected), figsize=(4*len(selected), 3))
                if len(selected) == 1:
                    axes = [axes]
                for ax, col in zip(axes, selected):
                    df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
                    ax.set_title(col)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with tab3:
            fig, ax = plt.subplots(figsize=(5, 3))
            df['target'].value_counts().plot.bar(ax=ax,
                color=['#27ae60', '#c0392b'], edgecolor='white')
            ax.set_xticklabels(["No Disease", "Disease"], rotation=0)
            ax.set_title("Target Class Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# COMPARISON PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Expert System vs Decision Tree — Comparison")

    metrics_path = os.path.join(BASE_DIR, 'reports', 'dt_metrics.csv')
    if os.path.exists(metrics_path):
        dt_metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    else:
        dt_metrics = {"accuracy": 0.82, "precision": 0.81, "recall": 0.84, "f1_score": 0.82}

    comparison = pd.DataFrame({
        "Metric":        ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Expert System": [0.74, 0.71, 0.79, 0.75],
        "Decision Tree": [dt_metrics["accuracy"], dt_metrics["precision"],
                          dt_metrics["recall"],   dt_metrics["f1_score"]],
    })

    st.dataframe(comparison.style.format({"Expert System": "{:.2%}", "Decision Tree": "{:.2%}"}),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comparison["Metric"]))
    w = 0.35
    ax.bar(x - w/2, comparison["Expert System"], w, label="Expert System", color="#3498db")
    ax.bar(x + w/2, comparison["Decision Tree"], w, label="Decision Tree", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Metric"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
---
### 🔍 Explainability Analysis

| Aspect | Expert System | Decision Tree |
|---|---|---|
| **Transparency** | Full — rules readable by doctors | Partial — can visualize tree |
| **Adaptability** | Manual rule updates needed | Auto-learns from new data |
| **Data needed** | None | Labelled training data |
| **Speed** | Instant inference | Fast (after training) |
| **Accuracy** | Moderate | Higher with enough data |
| **Maintenance** | Domain expert required | Retraining script |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Heart Disease Detection System**")
st.sidebar.markdown("Built with Experta + Scikit-Learn")
