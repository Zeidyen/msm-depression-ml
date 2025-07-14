
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
file_path = "Final_Processed_Dataset (1).csv"
df = pd.read_csv(file_path)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Define income bins and labels
income_bins = [0, 999, 4999, np.inf]
income_labels = ["Low", "Middle", "High"]
df["Income_Category"] = pd.cut(df["D5Income_monthly"], bins=income_bins, labels=income_labels, right=True)

# Define age bins and labels
age_bins = [0, 25, 40, 100]
age_labels = ["Young", "Middle-aged", "Older"]
df["Age_Group"] = pd.cut(df["D1Age"], bins=age_bins, labels=age_labels, right=True)

# Drop original age and income columns
df_binned = df.drop(columns=["D1Age", "D5Income_monthly"])

# One-hot encoding for categorical variables
df_binned = pd.get_dummies(df_binned, columns=["Income_Category", "Age_Group"], drop_first=True)

# Define selected features
selected_features_binned = [
    "D4EducLevel", "D7Sex_attract", "D8Married", "D9Religion",
    "PSS1", "PSS2", "PSS3", "PSS4", "PSS5", "PSS6", "PSS7", "PSS8", "PSS9", "PSS10",
    "PSS11", "PSS12", "PSS13", "PSS14",
    "ExtSocialIso1", "ExtSocialIso2", "ExtSocialIso3", "ExtSocialIso4", "ExtSocialIso5",
    "ExtSocialIso6", "ExtSocialIso7", "ExtSocialIso8",
    "IntSocialIso1", "IntSocialIso2", "IntSocialIso3", "IntSocialIso4", "IntSocialIso5",
    "IntSocialIso6", "IntSocialIso7", "IntSocialIso8", "IntSocialIso9", "IntSocialIso10",
    "IntSocialIso11", "IntSocialIso12", "IntSocialIso13", "BRScale1", "BRScale2", "BRScale3", "BRScale4", "BRScale5", "BRScale6",
    "StigmaSSB1", "StigmaSSB2", "StigmaSSB3", "StigmaSSB4", "StigmaSSB5", "StigmaSSB6", "StigmaSSB7", "StigmaSSB8", "StigmaSSB9", "StigmaSSB10",
    "StigmaGNC1", "StigmaGNC2", "StigmaGNC3", "StigmaGNC4", "StigmaGNC5", "StigmaGNC6", "StigmaGNC7", "StigmaGNC8", "StigmaGNC9", "StigmaGNC10", "StigmaGNC11", "StigmaGNC12", "StigmaGNC13",
    "SCS1", "SCS2", "SCS3", "SCS4", "SCS5", "SCS6", "SCS7", "SCS8",
    "Income_Category_Middle", "Income_Category_High",
    "Age_Group_Middle-aged", "Age_Group_Older"
]

# Extract features and target variable
label_encoder = LabelEncoder()
X_binned = df_binned[selected_features_binned]
y_binned = label_encoder.fit_transform(df_binned["Depression_Status"])

# Standardize numerical features
scaler = StandardScaler()
X_binned_scaled = scaler.fit_transform(X_binned)

# Split data into training and testing sets
X_train_binned, X_test_binned, y_train_binned, y_test_binned = train_test_split(
    X_binned_scaled, y_binned, test_size=0.2, random_state=42
)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_binned, y_train_binned)

# Define tree-based models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_balanced, y_train_balanced)

    # Predictions
    y_pred = model.predict(X_test_binned)
    y_pred_proba = model.predict_proba(X_test_binned)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test_binned, y_pred)
    precision = precision_score(y_test_binned, y_pred)
    recall = recall_score(y_test_binned, y_pred)
    f1 = f1_score(y_test_binned, y_pred)
    roc_auc = roc_auc_score(y_test_binned, y_pred_proba)
    conf_matrix = confusion_matrix(y_test_binned, y_pred)

    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Confusion Matrix": conf_matrix
    }

    # Print results
    print(f"Results for {name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC Score: {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{conf_matrix}")
    print("-" * 50)

# Compare Results
df_results = pd.DataFrame(results).T

# Plot results
plt.figure(figsize=(12, 6))
df_results[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].plot(kind='bar', figsize=(14, 6))
plt.title("Model Comparison - Tree-Based Classifiers")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.show()
