# Mood, Motivation & Behaviour Analysis for People with Schizophrenia

## Project Overview
This project contains machine learning models to analyze and diagnose mental disorders, specifically designed for analyzing mood, motivation, and behavioral patterns in individuals with schizophrenia and other mental health conditions. The system uses classification algorithms to predict mental disorder diagnoses based on patient data and provides personalized feedback.

---

## Code Files

### 1. **main.py** - Advanced Model Training with Hyperparameter Tuning
**Purpose:** Trains multiple classification models with optimized hyperparameters and uses ensemble learning.

**Key Features:**
- **Data Processing:**
  - Loads mental disorder dataset
  - Encodes categorical features using LabelEncoder
  - Applies SMOTE for handling class imbalance
  - Feature scaling with StandardScaler

- **Dimensionality Reduction:**
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)

- **Models Trained:**
  - Logistic Regression
  - Random Forest (with GridSearchCV)
  - Support Vector Machine - SVM (with GridSearchCV)
  - XGBoost (with GridSearchCV)
  - Stacking Ensemble (combines all three with LogisticRegression as final estimator)

- **Hyperparameter Tuning:**
  - Uses GridSearchCV with 3-fold cross-validation
  - Optimizes multiple parameters for each model

- **Evaluation Metrics:**
  - Accuracy Score
  - F1 Score
  - ROC-AUC Score
  - Classification Report

---

### 2. **code2.py** - Baseline Model Training with Feedback System
**Purpose:** Trains basic classification models and provides personalized patient feedback based on symptoms.

**Key Features:**
- **Data Processing:**
  - Loads and preprocesses mental disorder dataset
  - Encodes categorical features
  - Splits data into training and testing sets

- **Models Trained:**
  - Logistic Regression
  - Random Forest
  - SVM
  - XGBoost

- **Evaluation Metrics:**
  - Accuracy Score
  - F1 Score (weighted)
  - ROC-AUC Score
  - Classification Report

- **Model Persistence:**
  - Saves trained models as pickle files (.pkl)

- **Feedback Generation:**
  - `generate_feedback()` function provides personalized health recommendations
  - Based on specific patient symptoms:
    - Sleep disorders → Sleep regulation advice
    - Mood swings → Stress management recommendations
    - Suicidal thoughts → Immediate professional help alert
    - Sadness patterns → Mental health support suggestions

---

## Dataset
- **File:** `Dataset-Mental-Disorders.csv`
- **Target Column:** `Expert Diagnose` (mental disorder diagnosis)
- **Features:** Patient characteristics and behavioral/mood indicators
- **Key Columns:**
  - Patient Number (identifier)
  - Sleep disorder
  - Mood Swing
  - Suicidal thoughts
  - Sadness
  - Other clinical indicators

---

## Requirements
```
pandas
scikit-learn
xgboost
imbalanced-learn (imblearn)
matplotlib
seaborn
joblib
```

---

## How to Use

### 1. **Prepare Data**
   - Ensure `Dataset-Mental-Disorders.csv` is in the same directory as the scripts
   - Dataset should contain all required columns

### 2. **Run main.py (Advanced Training)**
   ```bash
   python main.py
   ```
   - Trains models with hyperparameter optimization
   - Compares stacking ensemble with individual models
   - Outputs performance metrics for all models

### 3. **Run code2.py (Basic Training + Feedback)**
   ```bash
   python code2.py
   ```
   - Trains four baseline models
   - Saves models as `.pkl` files
   - Generates personalized feedback for patients
   - Displays classification reports

---

## Key Differences Between Scripts

| Feature | main.py | code2.py |
|---------|---------|----------|
| **Hyperparameter Tuning** | ✅ GridSearchCV | ❌ Default parameters |
| **Dimensionality Reduction** | ✅ PCA + LDA | ❌ None |
| **Class Imbalance Handling** | ✅ SMOTE | ❌ None |
| **Ensemble Methods** | ✅ Stacking | ❌ Individual models |
| **Patient Feedback** | ❌ No | ✅ Yes |
| **Model Saving** | ❌ No | ✅ Yes (.pkl) |

---

## Output

### Models Generate:
1. **Accuracy & Performance Metrics** - Shows how well each model predicts diagnoses
2. **Classification Reports** - Detailed precision, recall, F1-scores per class
3. **Trained Model Files** (code2.py) - Saved as `.pkl` for future predictions
4. **Patient Feedback** (code2.py) - Personalized recommendations based on symptoms

---

## Clinical Application
This system can be used for:
- **Diagnostic Support:** Assist clinicians in identifying mental disorders
- **Patient Monitoring:** Track behavioral and mood patterns over time
- **Early Intervention:** Identify warning signs (e.g., suicidal thoughts)
- **Personalized Care:** Provide tailored health recommendations
- **Research:** Analyze patterns in mental health conditions

---

## Notes
- Adjust `test_size` in train_test_split for different data splits
- Modify model hyperparameters in GridSearchCV dictionaries for fine-tuning
- Ensure categorical columns are properly encoded before training
- ROC-AUC calculation uses 'ovo' (one-vs-one) for multi-class problems

---

## Future Improvements
- Add cross-validation for better generalization
- Implement feature importance analysis
- Add visualization dashboards
- Integrate with electronic health records (EHR)
- Deploy as a web application for clinical use
