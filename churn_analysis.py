import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load the Dataset
print("Loading dataset...")
df = pd.read_csv('Telco-Customer-Churn.csv')

# 2. Data Cleaning
print("\n--- Data Cleaning ---")
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop('customerID', axis=1, inplace=True)

# 3. Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')
plt.close()

# 4. Feature Engineering & Preprocessing
print("\n--- Feature Engineering & Preprocessing ---")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in df.columns if col not in numerical_cols + ['Churn']]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 5. Handling Class Imbalance
print("\n--- Handling Class Imbalance with SMOTE ---")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 6. Model Building & Hyperparameter Tuning
print("\n--- Model Tuning & Evaluation ---")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Tuning Logistic Regression
print("Tuning Logistic Regression using GridSearchCV...")
log_param_grid = {'C': [0.01, 0.1, 1, 10]}
log_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), log_param_grid, cv=3, scoring='roc_auc')
log_grid.fit(X_train_resampled, y_train_resampled)
best_log_model = log_grid.best_estimator_

# Tuning Random Forest
print("Tuning Random Forest using GridSearchCV...")
rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='roc_auc')
rf_grid.fit(X_train_resampled, y_train_resampled)
best_rf_model = rf_grid.best_estimator_

models = {
    "Logistic Regression (Tuned)": best_log_model,
    "Random Forest (Tuned)": best_rf_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n--- {name} Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'cm_{name.replace(" ", "_")}.png')
    plt.close()

# 7. Feature Importance
print("\n--- Feature Importance (Tuned Random Forest) ---")
importances = best_rf_model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15))
plt.title('Top 15 Most Important Features for Churn')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nAnalysis Complete! Models have been tuned and evaluated.")
