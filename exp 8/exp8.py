# ============================================================
# EXPERIMENT 8: CNN on MNIST Dataset
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
print("STEP 1: Loading Dataset...")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# -------------------------------
# STEP 2: Preprocessing
# -------------------------------
print("\nSTEP 2: Preprocessing...")

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -------------------------------
# STEP 3: Build Model
# -------------------------------
print("\nSTEP 3: Building CNN Model...")

model = Sequential([
    tf.keras.Input(shape=(28,28,1)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

model.summary()

# -------------------------------
# STEP 4: Compile Model
# -------------------------------
print("\nSTEP 4: Compiling Model...")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# STEP 5: Train Model
# -------------------------------
print("\nSTEP 5: Training Model...")

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# -------------------------------
# STEP 6: Evaluate Model
# -------------------------------
print("\nSTEP 6: Evaluating Model...")

loss, acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# -------------------------------
# STEP 7: Accuracy Graph
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.grid(True)
plt.show()

# -------------------------------
# STEP 8: Loss Graph
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])
plt.grid(True)
plt.show()
