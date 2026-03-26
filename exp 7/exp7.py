# ============================================================
# EXPERIMENT 7
# Artificial Neural Network (ANN) using TensorFlow/Keras
# Clean Professional Version
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

print("\n" + "="*60)
print("STEP 1: Loading Iris Dataset")
print("="*60)

# ------------------------------------------------------------
# STEP 1: Load Dataset
# ------------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Convert labels into categorical (One-Hot Encoding)
y = to_categorical(y)

print("Dataset Shape:", X.shape)
print("Number of Classes:", y.shape[1])


print("\n" + "="*60)
print("STEP 2: Train-Test Split")
print("="*60)

# ------------------------------------------------------------
# STEP 2: Split Dataset
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\n" + "="*60)
print("STEP 3: Feature Scaling")
print("="*60)

# ------------------------------------------------------------
# STEP 3: Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("\n" + "="*60)
print("STEP 4: Building ANN Model")
print("="*60)

# ------------------------------------------------------------
# STEP 4: Build ANN Model (Correct Input Layer)
# ------------------------------------------------------------
model = Sequential()

# Input Layer
model.add(Input(shape=(4,)))

# Hidden Layers
model.add(Dense(16, activation='relu'))
model.add(Dense(12, activation='relu'))

# Output Layer
model.add(Dense(3, activation='softmax'))

# Display Model Summary
model.summary()


print("\n" + "="*60)
print("STEP 5: Compile Model")
print("="*60)

# ------------------------------------------------------------
# STEP 5: Compile Model
# ------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


print("\n" + "="*60)
print("STEP 6: Training Model")
print("="*60)

# ------------------------------------------------------------
# STEP 6: Train Model
# ------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)


print("\n" + "="*60)
print("STEP 7: Evaluating Model")
print("="*60)

# ------------------------------------------------------------
# STEP 7: Evaluate Model
# ------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
print(f"Final Test Loss: {loss:.4f}")


print("\n" + "="*60)
print("STEP 8: Plotting Graphs")
print("="*60)

# ------------------------------------------------------------
# STEP 8: Accuracy Graph
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], marker='o')
plt.plot(history.history['val_accuracy'], marker='s')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n" + "-"*50 + "\n")

# ------------------------------------------------------------
# STEP 9: Loss Graph
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='s')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Training Loss', 'Validation Loss'])
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nExperiment Completed Successfully ✅")
