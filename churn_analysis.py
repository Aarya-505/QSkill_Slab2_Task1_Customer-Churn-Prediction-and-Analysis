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
print(f"Dataset Shape: {df.shape}")

# 2. Data Cleaning
print("\n--- Data Cleaning ---")
# Replace empty spaces with NaN in TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
# Convert TotalCharges to float
df['TotalCharges'] = df['TotalCharges'].astype(float)
# Fill missing values in TotalCharges with the median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
# Drop customerID as it is not a useful feature
df.drop('customerID', axis=1, inplace=True)
print("Missing values handled and customerID dropped.")

# 3. Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')
plt.close()
print("Saved churn_distribution.png")

# 4. Feature Engineering & Preprocessing
print("\n--- Feature Engineering & Preprocessing ---")
# Convert Target Variable to Binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify numerical and categorical columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in df.columns if col not in numerical_cols + ['Churn']]

# One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {df_encoded.shape}")

# Split into X and y
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 5. Handling Class Imbalance
print("\n--- Handling Class Imbalance with SMOTE ---")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training data shape after SMOTE: {X_train_resampled.shape}")

# 6. Model Building & Evaluation
print("\n--- Model Building ---")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"--- {name} Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'cm_{name.replace(" ", "_")}.png')
    plt.close()

# 7. Feature Importance
print("\n--- Feature Importance (Random Forest) ---")
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15))
plt.title('Top 15 Most Important Features for Churn')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Saved feature_importance.png")
print("\nTop 5 features driving churn:")
print(feature_imp_df.head(5))
print("\nAnalysis Complete! Please check the generated PNG files for visualizations.")
