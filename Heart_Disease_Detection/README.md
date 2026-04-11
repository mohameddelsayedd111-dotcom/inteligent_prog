# ❤️ Heart Disease Detection System

A hybrid AI system that detects heart disease risk using:
- **Rule-Based Expert System** (Experta) — 12 clinical rules
- **Decision Tree Classifier** (Scikit-Learn) — trained ML model
- **Streamlit Web UI** — interactive dashboard

---

## 📁 Project Structure

```
Heart_Disease_Detection/
├── data/
│   ├── raw_data.csv          # Original dataset (303 patients, 14 features)
│   └── cleaned_data.csv      # Preprocessed & normalized dataset
├── notebooks/
│   ├── data_analysis.ipynb   # EDA, visualizations, insights
│   └── model_training.ipynb  # End-to-end training walkthrough
├── rule_based_system/
│   ├── rules.py              # 12 Experta rules + KnowledgeEngine
│   └── expert_system.py      # Inference engine + interactive CLI
├── ml_model/
│   ├── train_model.py        # Decision Tree training + evaluation
│   ├── predict.py            # Load model + single/batch prediction
│   └── heart_disease_model.pkl  # Saved model (after training)
├── utils/
│   └── data_processing.py    # Preprocessing pipeline
├── reports/
│   ├── accuracy_comparison.md # Expert System vs Decision Tree report
│   ├── dt_metrics.csv         # Saved ML metrics
│   ├── feature_importance.png # Feature ranking chart
│   └── confusion_matrix.png   # Confusion matrix heatmap
├── ui/
│   └── app.py                # Streamlit dashboard
├── README.md
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data & Train the Model

```bash
cd ml_model
python train_model.py
```

This will:
- Clean and normalize `data/raw_data.csv` → `data/cleaned_data.csv`
- Train and tune a Decision Tree Classifier (GridSearchCV)
- Save the model to `ml_model/heart_disease_model.pkl`
- Generate charts in `reports/`

### 3. Run the Expert System (CLI)

```bash
cd rule_based_system
python expert_system.py
```

Enter patient vitals when prompted to receive a rule-based risk assessment.

### 4. Launch the Streamlit Dashboard

```bash
cd ui
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Expert System Rules

The expert system contains **12 clinical rules**:

| # | Condition | Risk Change |
|---|-----------|:-----------:|
| R01 | Cholesterol > 240 AND Age > 50 | +30 |
| R02 | Blood Pressure > 140 AND Exercise Angina | +35 |
| R03 | Chest pain (typical/atypical type) | +25 |
| R04 | ≥ 2 major vessels blocked | +35 |
| R05 | ST Depression > 2.0 | +30 |
| R06 | High fasting blood sugar AND Age > 45 | +20 |
| R07 | Max heart rate < 120 bpm | +20 |
| R08 | Male + Age > 55 + Cholesterol > 200 | +25 |
| R09 | Exercise angina + ST depression > 1.5 | +30 |
| R10 | Severe hypertension > 160 mmHg | +35 |
| R11 | High HR ≥ 150 + Normal cholesterol ≤ 200 | -15 |
| R12 | No angina + Low ST + 0 blocked vessels | -20 |

**Risk Classification:**
- Score < 25 → 🟢 LOW
- 25 ≤ Score < 60 → 🟡 MODERATE  
- Score ≥ 60 → 🔴 HIGH

---

## 📊 Dataset Features

| Feature    | Description |
|------------|-------------|
| age        | Age in years |
| sex        | 0=Female, 1=Male |
| cp         | Chest pain type (0–3) |
| trestbps   | Resting blood pressure (mmHg) |
| chol       | Serum cholesterol (mg/dl) |
| fbs        | Fasting blood sugar > 120 mg/dl (0/1) |
| restecg    | Resting ECG results (0–2) |
| thalach    | Maximum heart rate achieved |
| exang      | Exercise induced angina (0/1) |
| oldpeak    | ST depression induced by exercise |
| slope      | Slope of peak exercise ST segment |
| ca         | Number of major vessels (0–3) |
| thal       | Thalassemia (1–3) |
| **target** | **Heart disease (0=No, 1=Yes)** |

---

## 📈 Model Performance (Approximate)

| Metric    | Expert System | Decision Tree |
|-----------|:---:|:---:|
| Accuracy  | 74% | ~83% |
| Precision | 71% | ~82% |
| Recall    | 79% | ~85% |
| F1-Score  | 75% | ~83% |

See `reports/accuracy_comparison.md` for the full analysis.

---

## 🛠️ Technology Stack

- **Python 3.10+**
- **Pandas / NumPy** — Data processing
- **Scikit-Learn** — Decision Tree + evaluation
- **Experta** — Rule-based inference engine
- **Matplotlib / Seaborn** — Visualizations
- **Streamlit** — Interactive web dashboard
- **Joblib** — Model persistence

---

*Expert Systems Project — Heart Disease Detection*
