# =====================================
# IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# LOAD DATASET
# =====================================
df = pd.read_csv("/content/sample_data/Titanic-Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =====================================
# DATA PREPROCESSING
# =====================================

# Drop unnecessary columns
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missing values (CORRECT METHOD)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical columns
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# DECISION TREE
# =====================================
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# =====================================
# DECISION TREE VISUALIZATION
# =====================================
plt.figure(figsize=(15,8))
plot_tree(dt,
          feature_names=X.columns,
          class_names=["Not Survived", "Survived"],
          filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# =====================================
# RANDOM FOREST
# =====================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# =====================================
# ACCURACY COMPARISON GRAPH
# =====================================
models = ["Decision Tree", "Random Forest"]
accuracies = [
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_rf)
]

plt.figure()
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# =====================================
# FEATURE IMPORTANCE (Random Forest)
# =====================================
importances = rf.feature_importances_

plt.figure()
plt.bar(X.columns, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.show()
