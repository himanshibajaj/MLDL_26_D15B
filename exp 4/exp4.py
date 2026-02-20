# ============================================================
# EXP 4: KNN with Full Process + Evaluation + Tuning
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("/content/sample_data/heart.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ------------------------------------------------------------
# STEP 2: Define Features and Target
# ------------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------------------------------------
# STEP 3: Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# STEP 4: Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# STEP 5: Train Initial KNN Model
# ------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\n===== INITIAL KNN RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Confusion Matrix Heatmap
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------------------------
# STEP 6: Hyperparameter Tuning
# ------------------------------------------------------------
param_grid = {
    "n_neighbors": range(1, 31),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\n===== HYPERPARAMETER TUNING =====")
print("Best Parameters:", grid.best_params_)
print("Best Cross Validation Score:", grid.best_score_)

# ------------------------------------------------------------
# STEP 7: Evaluate Tuned Model
# ------------------------------------------------------------
best_knn = grid.best_estimator_
y_pred_best = best_knn.predict(X_test)

print("\n===== TUNED KNN RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_best))

# ------------------------------------------------------------
# STEP 8: K vs Accuracy Graph
# ------------------------------------------------------------
k_range = range(1, 31)
accuracy_scores = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))

plt.figure(figsize=(8,5))
plt.plot(k_range, accuracy_scores)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()
