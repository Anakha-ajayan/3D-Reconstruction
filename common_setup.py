import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\train_set_final.csv")

# Step 2: Features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for future use
os.makedirs("saved_models_final", exist_ok=True)
joblib.dump(scaler, "saved_models_final/standard_scaler.pkl")

# Step 4: Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 5: Create mean_fpr for ROC curve averaging
mean_fpr = np.linspace(0, 1, 100)
