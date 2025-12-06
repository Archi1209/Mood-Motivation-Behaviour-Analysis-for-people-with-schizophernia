# Paste your code here

# train_models_best_v2.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1. Load dataset
data_path = 'Dataset-Mental-Disorders.csv'
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 6. Standardize features for PCA / LDA / SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Apply PCA + LDA
try:
    pca = PCA(n_components=min(X_train_scaled.shape[1], 10))
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    lda = LDA(n_components=min(len(set(y_train))-1, X_train_pca.shape[1]))
    X_train_lda = lda.fit_transform(X_train_pca, y_train)
    X_test_lda = lda.transform(X_test_pca)
except:
    X_train_lda, X_test_lda = X_train_scaled, X_test_scaled

# 8. Define models with hyperparameter tuning
rf_params = {'n_estimators':[100,200], 'max_depth':[None,5,10]}
xgb_params = {'n_estimators':[100,200], 'max_depth':[3,5], 'learning_rate':[0.1,0.01]}
svm_params = {'C':[0.5,1,5], 'kernel':['linear','rbf'], 'gamma':['scale','auto']}

rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3)
xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), xgb_params, cv=3)
svm = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=3)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": rf,
    "SVM": svm,
    "XGBoost": xgb
}

# 9. Stacking Ensemble
base_models = [
    ('rf', rf),
    ('xgb', xgb),
    ('svm', svm)
]
stacking_model = StackingClassifier(
    estimators=base_models, final_estimator=LogisticRegression(), cv=5
)
models["Stacking Ensemble"] = stacking_model

# 10. Train, evaluate, collect results
comparison_data = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['SVM', 'Logistic Regression']:
        model.fit(X_train_lda, y_train)
        y_pred = model.predict(X_test_lda)
        try:
            y_prob = model.predict_proba(X_test_lda)
        except:
            y_prob = None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)
        except:
            y_prob = None

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovo') if y_prob is not None else None
    except:
        auc = None

    comparison_data.append([name, accuracy, f1, auc])
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc}")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))
    joblib.dump(model, f"{name.replace(' ','_')}_model.pkl")

# 11. Comparison matrix
comparison_matrix = pd.DataFrame(comparison_data, columns=['Model','Accuracy','F1 Score','AUC'])
high_acc = comparison_matrix[comparison_matrix['Accuracy']>=0.9]
if high_acc.empty:
    print("\nShowing all models instead.")
    display_matrix = comparison_matrix
else:
    display_matrix = high_acc

print("\n===== Model Comparison Matrix =====\n")
print(display_matrix.to_string(index=False))

# 12. Best model
best_model_row = comparison_matrix.loc[comparison_matrix['Accuracy'].idxmax()]
print("\n===== BEST MODEL =====")
print(f"Model: {best_model_row['Model']}")
print(f"Accuracy: {best_model_row['Accuracy']:.4f}")
print(f"F1 Score: {best_model_row['F1 Score']:.4f}")
print(f"AUC: {best_model_row['AUC']}")

# 13. Visual comparison
plt.figure(figsize=(10,5))
plt.bar(display_matrix['Model'], display_matrix['Accuracy'], color='skyblue')
plt.ylim(0,1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(10,5))
auc_values = [a if a is not None else 0 for a in display_matrix['AUC']]
plt.bar(display_matrix['Model'], auc_values, color='salmon')
plt.ylim(0,1)
plt.title('Model AUC Comparison (0 if not computable)')
plt.ylabel('AUC')
plt.show()