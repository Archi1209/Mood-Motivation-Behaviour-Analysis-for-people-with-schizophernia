# Paste your code here
# train_models_console_feedback_visual.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. Load dataset
data_path = 'Dataset-Mental-Disorders.csv'  # update path if needed
data = pd.read_csv(data_path)

# 2. Features and Target
target_column = 'Expert Diagnose'
X = data.drop(['Patient Number', target_column], axis=1)
y = data[target_column]

# 3. Encode categorical features
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(y)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 6. Train, evaluate, and collect results
comparison_data = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovo')
    except:
        auc = None
    
    comparison_data.append([name, accuracy, f1, auc])
    
    print(f"\n{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred, target_names=target_le.classes_)}")
    
    joblib.dump(model, f"{name.replace(' ', '_')}_model.pkl")

# 7. Show comparison matrix
#comparison_matrix = pd.DataFrame(comparison_data, columns=['Model', 'Accuracy', 'F1 Score', 'AUC'])
#print("\n===== Model Comparison Matrix =====\n")
#print(comparison_matrix.to_string(index=False))

# 8. Best model
#best_model_idx = comparison_matrix['F1 Score'].idxmax()
#best_model = comparison_matrix.iloc[best_model_idx]
#print("\n===== BEST MODEL =====\n")
#print(f"Model: {best_model['Model']}")
#print(f"Accuracy: {best_model['Accuracy']:.4f}")
#print(f"F1 Score: {best_model['F1 Score']:.4f}")
#print(f"AUC: {best_model['AUC']}")

# 9. Feedback function
def generate_feedback(patient_row):
    feedback = []
    for col in patient_row:
        val = patient_row[col]
        if col == 'Sleep dissorder' and val in ['Most-Often', 'Usually']:
            feedback.append("Regulate your sleep, avoid oversleeping.")
        elif col == 'Mood Swing' and val == 'YES':
            feedback.append("Consider stress management to stabilize mood swings.")
        elif col == 'Suicidal thoughts' and val == 'YES':
            feedback.append("Seek professional help immediately!")
        elif col == 'Sadness' and val in ['Most-Often', 'Usually']:
            feedback.append("Engage in positive activities to improve mood.")
        elif col == 'Euphoric' and val in ['Most-Often', 'Usually']:
            feedback.append("Monitor excessive euphoria, avoid risky behavior.")
        elif col == 'Exhausted' and val in ['Most-Often', 'Usually']:
            feedback.append("Ensure adequate rest to reduce exhaustion.")
        elif col == 'Anorxia' and val == 'YES':
            feedback.append("Maintain proper nutrition.")
        elif col == 'Authority Respect' and val == 'NO':
            feedback.append("Improve communication and respect towards authority.")
    return feedback

# 10. Generate feedback and visualization for each test sample
print("\n===== Feedback for Test Samples =====\n")
for idx, row in X_test.iterrows():
    # Convert encoded features back to original using label_encoders
    patient_data = {}
    for col in X_test.columns:
        if col in categorical_cols:
            le = label_encoders[col]
            patient_data[col] = le.inverse_transform([row[col]])[0]
        else:
            patient_data[col] = row[col]
    
    fb = generate_feedback(patient_data)
    
    print(f"Patient {data.loc[idx,'Patient Number']} Feedback:")
    if fb:
        for f in fb:
            print(" -", f)
    else:
        print(" - No critical feedback, patient data looks normal.")
    
    # Visualization: show patient's behavioral profile
    plt.figure(figsize=(10,5))
    behaviors = list(patient_data.keys())
    values = []
    for col in behaviors:
        val = patient_data[col]
        if isinstance(val, str):
            # Map categories to numeric for plotting
            if val in ['Seldom', 'Sometimes', 'Usually', 'Most-Often']:
                mapping = {'Seldom':1, 'Sometimes':2, 'Usually':3, 'Most-Often':4}
                values.append(mapping[val])
            elif val in ['NO', 'YES']:
                mapping = {'NO':0, 'YES':1}
                values.append(mapping[val])
            else:
                values.append(0)
        else:
            values.append(val)
    
    sns.barplot(x=behaviors, y=values, palette='coolwarm')
    plt.title(f"Patient {data.loc[idx,'Patient Number']} Behavioral Profile")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Behavior Level")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()
