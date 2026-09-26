#!/usr/bin/env python
# coding: utf-8

# # Predictive Maintenance — Failure Risk Modeling
# 
# **Objective:** identify machine observations associated with failure in the next 30 days and explain the operational factors behind the predictions.
# 
# **Workflow:** source reconciliation → feature engineering → class-imbalance handling → stratified cross-validation → hyperparameter tuning → model explanation → high-risk predictions.
# 
# All project data is synthetic and contains no PII.

# In[1]:


# 1. Data preparation — prebuilt working block
import pandas as pd
import numpy as np

machines = pd.read_csv("machine_metadata.csv")
readings = pd.read_csv("machine_sensor_readings.csv")
maintenance = pd.read_csv("maintenance_history.csv")

readings["timestamp"] = pd.to_datetime(readings["timestamp"])
maintenance["maintenance_date"] = pd.to_datetime(maintenance["maintenance_date"])

# Reconcile overlapping reference fields: machine_metadata is the reference source.
readings = readings.drop(columns=["plant", "machine_type", "machine_age_years", "rated_load_pct"])
df = readings.merge(machines, on="machine_id", how="left")

maintenance_summary = (
    maintenance.groupby("machine_id")
    .agg(
        maintenance_count=("maintenance_date", "count"),
        corrective_count=("maintenance_type", lambda s: (s == "Corrective").sum()),
        avg_maintenance_duration=("maintenance_duration_hr", "mean")
    ).reset_index()
)

df = df.merge(maintenance_summary, on="machine_id", how="left")
df[["maintenance_count","corrective_count","avg_maintenance_duration"]] = (
    df[["maintenance_count","corrective_count","avg_maintenance_duration"]].fillna(0)
)

print(f"Modeling rows: {len(df):,}")
print(f"Failure rate: {df['failure_next_30d'].mean():.1%}")
display(df.head())


# ## Analytical decision 1 — Feature engineering
# 
# Modify one short line to add a derived operational stress feature, then include it in the feature list. The feature should have a defensible relationship to machine failure risk.

# In[3]:


# 2. Feature engineering — meaningful edit required
df["thermal_stress"] = (df["temperature_c"]-70) * (df["load_pct"]/100)

features = [
    "temperature_c", "vibration_mm_s", "pressure_kpa", "load_pct",
    "machine_age_years", "operating_hours", "maintenance_count",
    "corrective_count", "thermal_stress"
]

X = df[features].copy()
y = df["failure_next_30d"]

print("Features:", features)
display(y.value_counts(normalize=True).rename("proportion"))


# In[4]:


display(df[["temperature_c", "load_pct", "thermal_stress"]].head())


# ## Analytical decision 2 — Imbalance-aware modeling and validation
# 
# The failure class is intentionally uncommon. Compare models using stratified cross-validation and focus on recall/F1 as well as ROC-AUC. The model configuration contains a short parameter that can be modified during the task.

# In[6]:


# 3. Baseline vs stronger model with stratified cross-validation
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

strong_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

scoring = {"precision":"precision", "recall":"recall", "f1":"f1", "roc_auc":"roc_auc", "pr_auc":"average_precision"}
results = {}
for name, model in {"Logistic baseline": baseline, "Random Forest": strong_model}.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {m: scores[f"test_{m}"].mean() for m in scoring}

comparison = pd.DataFrame(results).T.sort_values("f1", ascending=False)
display(comparison.round(3))


# ## Analytical decision 3 — Hyperparameter tuning
# 
# Refine the small search space based on the cross-validation results. Avoid a large brute-force search; the goal is to make a defensible complexity decision.

# In[11]:


# 4. Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    baseline, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1
)
grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print("Best cross-validated F1:", round(grid.best_score_, 3))


# ## Model explanation and risk output
# 
# Use permutation importance to explain the selected model and produce a ranked high-risk prediction artifact for maintenance prioritization.

# In[12]:


# 5. Explain the selected model and create the final artifact
from sklearn.inspection import permutation_importance
from pathlib import Path

final_model = grid.best_estimator_
final_model.fit(X, y)

perm = permutation_importance(
    final_model, X, y, scoring="f1", n_repeats=5, random_state=42, n_jobs=-1
)

importance = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
display(importance.head(8).rename("permutation_importance").to_frame())

risk_df = df[["timestamp","machine_id","plant","machine_type"]].copy()
risk_df["failure_probability"] = final_model.predict_proba(X)[:, 1]
risk_df = risk_df.sort_values("failure_probability", ascending=False).head(20)

display(risk_df)

Path("outputs").mkdir(exist_ok=True)
risk_df.to_csv("outputs/high_risk_machine_predictions.csv", index=False)
print("Saved: outputs/high_risk_machine_predictions.csv")


# In[ ]:




