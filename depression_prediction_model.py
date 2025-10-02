# ==============================
# Depression ML Study – Full Script
# (Main analysis + 3-predictor sensitivity analysis)
# Leakage-safe pipelines; SMOTE in-train only
# ==============================

# --- Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, RocCurveDisplay,
                             PrecisionRecallDisplay)
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ------------------------------
# Config
# ------------------------------
file_path = "Final_Processed_Dataset (1).csv"  # change if needed
RANDOM_STATE = 42

# ------------------------------
# Helpers
# ------------------------------
def warn(msg):
    print(f"[WARN] {msg}")

def find_exact(df, name):
    m = {c.lower(): c for c in df.columns}
    return m.get(name.lower())

def find_contains(df, substr):
    for c in df.columns:
        if substr.lower() in c.lower():
            return c
    return None

def safe_get_existing(df, wanted_cols):
    existing = [c for c in wanted_cols if c in df.columns]
    missing = sorted(set(wanted_cols) - set(existing))
    if missing:
        warn(f"Missing features skipped: {missing}")
    return existing

def pos_label_from_classes(classes):
    # Prefer a class name containing 'depress'; else if binary, use index 1
    for i, k in enumerate(classes):
        if "depress" in str(k).lower():
            return i
    return 1 if len(classes) == 2 else None

# ------------------------------
# Load data
# ------------------------------
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find: {file_path}")

df = pd.read_csv(file_path)
df = df.copy()
df.fillna(df.median(numeric_only=True), inplace=True)

# ------------------------------
# Create bins for Income & Age (if present)
# ------------------------------
if "D5Income_monthly" in df.columns:
    income_bins = [0, 999, 4999, np.inf]
    income_labels = ["Low", "Middle", "High"]
    df["Income_Category"] = pd.cut(df["D5Income_monthly"], bins=income_bins,
                                   labels=income_labels, right=True, include_lowest=True)
else:
    warn("D5Income_monthly not found — Income_Category will not be created.")

if "D1Age" in df.columns:
    age_bins = [0, 25, 40, 100]
    age_labels = ["Young", "Middle-aged", "Older"]
    df["Age_Group"] = pd.cut(df["D1Age"], bins=age_bins, labels=age_labels,
                             right=True, include_lowest=True)
else:
    warn("D1Age not found — Age_Group will not be created.")

drop_cols = [c for c in ["D1Age", "D5Income_monthly"] if c in df.columns]
df_binned = df.drop(columns=drop_cols)

# One-hot encode new bins (if they exist)
for cat_col in ["Income_Category", "Age_Group"]:
    if cat_col in df_binned.columns:
        df_binned = pd.get_dummies(df_binned, columns=[cat_col], drop_first=True)

# ------------------------------
# Feature set (main analysis)
# ------------------------------
selected_features_binned = [
    "D4EducLevel", "D7Sex_attract", "D8Married", "D9Religion",
    "PSS1","PSS2","PSS3","PSS4","PSS5","PSS6","PSS7","PSS8","PSS9","PSS10",
    "PSS11","PSS12","PSS13","PSS14",
    "ExtSocialIso1","ExtSocialIso2","ExtSocialIso3","ExtSocialIso4","ExtSocialIso5","ExtSocialIso6","ExtSocialIso7","ExtSocialIso8",
    "IntSocialIso1","IntSocialIso2","IntSocialIso3","IntSocialIso4","IntSocialIso5","IntSocialIso6","IntSocialIso7","IntSocialIso8",
    "IntSocialIso9","IntSocialIso10","IntSocialIso11","IntSocialIso12","IntSocialIso13",
    "BRScale1","BRScale2","BRScale3","BRScale4","BRScale5","BRScale6",
    "StigmaSSB1","StigmaSSB2","StigmaSSB3","StigmaSSB4","StigmaSSB5","StigmaSSB6","StigmaSSB7","StigmaSSB8","StigmaSSB9","StigmaSSB10",
    "StigmaGNC1","StigmaGNC2","StigmaGNC3","StigmaGNC4","StigmaGNC5","StigmaGNC6","StigmaGNC7","StigmaGNC8","StigmaGNC9","StigmaGNC10","StigmaGNC11","StigmaGNC12","StigmaGNC13",
    "SCS1","SCS2","SCS3","SCS4","SCS5","SCS6","SCS7","SCS8",
    "Income_Category_Middle","Income_Category_High",
    "Age_Group_Middle-aged","Age_Group_Older"
]

existing_main = safe_get_existing(df_binned, selected_features_binned)
X = df_binned[existing_main].copy()

# Target
target_col = find_exact(df_binned, "Depression_Status") or find_contains(df_binned, "Depress")
if target_col is None:
    raise KeyError("Target 'Depression_Status' (or a column containing 'depress') not found.")
le = LabelEncoder()
y = le.fit_transform(df_binned[target_col].astype(str))
print("Target classes:", dict(zip(le.classes_, range(len(le.classes_)))))
pos_label = pos_label_from_classes(le.classes_)

# ------------------------------
# Train/test split (stratified)
# ------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------
# Models (fixed params = baseline, no tuning)
# ------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                             eval_metric='logloss', n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
    "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=RANDOM_STATE)
}

def make_pipe(model):
    # Scaler -> SMOTE -> Model; scaler/SMOTE fit only on training folds inside CV
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model)
    ])

# ------------------------------
# Cross-validation + Test evaluation
# ------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results_main = {}

for name, base in models.items():
    pipe = make_pipe(base)

    # CV AUC on training (leakage-safe)
    try:
        cv_auc = cross_val_score(pipe, X_tr, y_tr, scoring="roc_auc", cv=cv, n_jobs=-1)
        cv_auc_mean, cv_auc_sd = float(np.mean(cv_auc)), float(np.std(cv_auc))
    except Exception as e:
        cv_auc_mean, cv_auc_sd = np.nan, np.nan
        warn(f"{name}: CV failed: {e}")

    # Fit and test
    pipe.fit(X_tr, y_tr)

    # Proba/scores for AUC
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        idx = pos_label if pos_label is not None else (1 if len(le.classes_)==2 else 0)
        y_proba = pipe.predict_proba(X_te)[:, idx]
    elif hasattr(pipe.named_steps["model"], "decision_function"):
        y_proba = pipe.decision_function(X_te)
    else:
        y_proba = pipe.predict(X_te)  # fallback

    y_pred = pipe.predict(X_te)

    avg = "binary" if pos_label is not None else "macro"
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average=avg, zero_division=0, pos_label=pos_label)
    rec  = recall_score(y_te, y_pred, average=avg, zero_division=0, pos_label=pos_label)
    f1   = f1_score(y_te, y_pred, average=avg, zero_division=0, pos_label=pos_label)
    auc  = roc_auc_score(y_te, y_proba)
    cm   = confusion_matrix(y_te, y_pred)

    results_main[name] = {
        "CV ROC AUC (mean±sd)": f"{cv_auc_mean:.3f}±{cv_auc_sd:.3f}",
        "Test Accuracy": acc, "Test Precision": prec, "Test Recall": rec,
        "Test F1": f1, "Test ROC AUC": auc, "Confusion Matrix": cm
    }

# Tidy table + plot
df_main = (pd.DataFrame(results_main).T
           .assign(**{
               'Test Accuracy': lambda d: d['Test Accuracy'].astype(float),
               'Test Precision': lambda d: d['Test Precision'].astype(float),
               'Test Recall': lambda d: d['Test Recall'].astype(float),
               'Test F1': lambda d: d['Test F1'].astype(float),
               'Test ROC AUC': lambda d: d['Test ROC AUC'].astype(float),
           }))
print("\n=== Main Analysis Summary ===\n", df_main[['CV ROC AUC (mean±sd)','Test Accuracy','Test Precision','Test Recall','Test F1','Test ROC AUC']])

ax = df_main[['Test Accuracy','Test Precision','Test Recall','Test F1','Test ROC AUC']].plot(
    kind='bar', figsize=(12, 6), rot=45
)
ax.set_title("Model Comparison – Tree-Based Classifiers (Leakage-free, Test Set)")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Optionally save
df_main.to_csv("main_model_results.csv")

# ==============================
# Reduced-feature (three-predictor) Sensitivity Analysis
# ==============================

# Prefer original items; fall back to domain-level proxies if needed
col_iso2 = find_exact(df_binned, "ExtSocialIso2") or find_contains(df_binned, "ExtSocialIso2")
col_trust = find_exact(df_binned, "IntSocialIso13") or find_contains(df_binned, "IntSocialIso13")

pss_items = []
for i in range(1, 15):
    c = find_exact(df_binned, f"PSS{i}") or find_contains(df_binned, f"PSS{i}")
    if c is not None:
        pss_items.append(c)

# If items missing, try proxies that sometimes appear in alternate files
proxy_iso = find_exact(df_binned, "ExtSocialIso_Level") or find_contains(df_binned, "ExtSocialIso_Level")
proxy_trust = find_exact(df_binned, "IntSocialIso_Level") or find_contains(df_binned, "IntSocialIso_Level")
proxy_pss = find_exact(df_binned, "PSS_StressLevel") or find_contains(df_binned, "PSS_StressLevel")

use_proxy = False
if (col_iso2 is None or col_trust is None or len(pss_items) < 3):
    warn("Raw items not fully available; using domain-level proxies for the 3-feature analysis.")
    use_proxy = True

if not use_proxy:
    df_binned["PSS_total"] = df_binned[pss_items].sum(axis=1)
    X3 = df_binned[[col_iso2, "PSS_total", col_trust]].copy()
    three_feature_names = [col_iso2, "PSS_total", col_trust]
else:
    needed = [proxy_pss, proxy_iso, proxy_trust]
    if any(v is None for v in needed):
        raise KeyError("Neither raw items nor proxies for PSS/External/Internal social isolation were found.")
    X3 = df_binned[[proxy_pss, proxy_iso, proxy_trust]].copy()
    # These are categorical proxies → we will OneHot-encode in the pipeline
    three_feature_names = [proxy_pss, proxy_iso, proxy_trust]

# Build y (same target, same encoder)
y3 = y.copy()

# Split
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3, y3, test_size=0.20, stratify=y3, random_state=RANDOM_STATE
)

# Two models: Logistic (interpretable baseline) and RF (top performer family)
if use_proxy:
    # One-hot needed for categorical proxies
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), list(range(X3.shape[1])))], remainder="drop")
    logit_pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RANDOM_STATE)),
                              ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    rf_pipe    = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RANDOM_STATE)),
                              ("clf", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE))])
else:
    # Continuous-ish inputs; scale then SMOTE
    logit_pipe = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=RANDOM_STATE)),
                              ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    rf_pipe    = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=RANDOM_STATE)),
                              ("clf", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE))])

models_3 = {
    "Logistic (3 predictors)": logit_pipe,
    "Random Forest (3 predictors)": rf_pipe
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []

for name, pipe in models_3.items():
    # CV AUC
    try:
        cv_auc = cross_val_score(pipe, X3_tr, y3_tr, scoring="roc_auc", cv=cv)
        cv_mean, cv_sd = float(np.mean(cv_auc)), float(np.std(cv_auc))
    except Exception as e:
        cv_mean, cv_sd = np.nan, np.nan
        warn(f"3-feature {name}: CV failed: {e}")

    # Fit & test
    pipe.fit(X3_tr, y3_tr)

    if hasattr(pipe.named_steps[list(pipe.named_steps.keys())[-1]], "predict_proba"):
        clf_key = list(pipe.named_steps.keys())[-1]
        idx = pos_label if pos_label is not None else (1 if len(le.classes_)==2 else 0)
        proba = pipe.named_steps[clf_key].predict_proba(pipe[:-1].transform(X3_te))[:, idx]
        # Note: for imblearn pipeline, calling pipe.predict_proba(X) is fine too; expanded here for clarity
        proba = pipe.predict_proba(X3_te)[:, idx]
    else:
        proba = pipe.decision_function(X3_te) if hasattr(pipe.named_steps[list(pipe.named_steps.keys())[-1]], "decision_function") else pipe.predict(X3_te)

    pred = (proba >= 0.5).astype(int) if proba.ndim == 1 else pipe.predict(X3_te)

    rows.append({
        "Model": name,
        "Features used": ", ".join(three_feature_names),
        "CV ROC AUC (mean)": cv_mean,
        "CV ROC AUC (sd)": cv_sd,
        "Test ROC AUC": roc_auc_score(y3_te, proba),
        "Test F1": f1_score(y3_te, pred, zero_division=0, pos_label=pos_label),
        "Test Recall": recall_score(y3_te, pred, zero_division=0, pos_label=pos_label),
        "Test Precision": precision_score(y3_te, pred, zero_division=0, pos_label=pos_label),
        "Test Accuracy": accuracy_score(y3_te, pred),
        "Confusion Matrix": [int(x) for x in confusion_matrix(y3_te, pred).ravel().tolist()]
    })

df_three = pd.DataFrame(rows)
print("\n=== Three-Predictor Sensitivity Summary ===\n", df_three)

# Plots: ROC & PR for the two 3-predictor models
plt.figure(figsize=(6,5))
for name, pipe in models_3.items():
    if hasattr(pipe.named_steps[list(pipe.named_steps.keys())[-1]], "predict_proba"):
        idx = pos_label if pos_label is not None else (1 if len(le.classes_)==2 else 0)
        prob_disp = pipe.predict_proba(X3_te)[:, idx]
    elif hasattr(pipe.named_steps[list(pipe.named_steps.keys())[-1]], "decision_function"):
        prob_disp = pipe.decision_function(X3_te)
    else:
        prob_disp = pipe.predict(X3_te)
    RocCurveDisplay.from_predictions(y3_te, prob_disp, name=name)
plt.plot([0,1],[0,1], linestyle="--")
plt.title("ROC – Three-Predictor Models (Test)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
for name, pipe in models_3.items():
    if hasattr(pipe.named_steps[list(pipe.named_steps.keys())[-1]], "predict_proba"):
        idx = pos_label if pos_label is not None else (1 if len(le.classes_)==2 else 0)
        prob_disp = pipe.predict_proba(X3_te)[:, idx]
    elif hasattr(pipe.named_steps[list(pipe.named_steps.keys())[-1]], "decision_function"):
        prob_disp = pipe.decision_function(X3_te)
    else:
        prob_disp = pipe.predict(X3_te)
    PrecisionRecallDisplay.from_predictions(y3_te, prob_disp, name=name)
plt.title("Precision–Recall – Three-Predictor Models (Test)")
plt.tight_layout()
plt.show()

# Save sensitivity results
df_three.to_csv("three_predictor_sensitivity_results.csv", index=False)

# ==============================
# End of script
# ==============================
