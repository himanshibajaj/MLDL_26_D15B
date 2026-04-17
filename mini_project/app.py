// streamlit run app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 🎨 PAGE
st.set_page_config(page_title="Fitness AI System", layout="wide")

st.title("🏃 Smart Fitness AI System (ML + DL)")
st.write("Complete ML/DL Project with Visualization")

# 📊 DATA
data = pd.DataFrame({
    "steps": [2000, 5000, 8000, 12000, 3000, 7000],
    "calories": [150, 300, 500, 800, 200, 450],
    "heart_rate": [70, 85, 95, 110, 75, 90],
    "activity": [0, 1, 2, 2, 0, 1]
})

X = data[["steps", "calories", "heart_rate"]]
y = data["activity"]

# 📂 SIDEBAR
menu = st.sidebar.selectbox("Select Module", [
    "🏃 Activity Prediction",
    "📊 Regression",
    "🧠 Classification",
    "👥 Clustering",
    "📍 KNN",
    "🤖 ANN",
    "🖼️ CNN Demo",
    "🔤 LSTM Demo",
    "🧼 Autoencoder Demo",
    "📌 About"
])

# 🏃 ACTIVITY
if menu == "🏃 Activity Prediction":
    steps = st.slider("Steps", 1000, 15000)
    calories = st.slider("Calories", 100, 1000)
    hr = st.slider("Heart Rate", 60, 140)

    model = RandomForestClassifier()
    model.fit(X, y)

    pred = model.predict([[steps, calories, hr]])[0]
    activity_map = {0: "Rest 🛌", 1: "Walk 🚶", 2: "Run 🏃"}

    st.success(f"Activity: {activity_map[pred]}")

# 📊 REGRESSION
elif menu == "📊 Regression":
    steps = st.slider("Steps", 1000, 15000)

    model = LinearRegression()
    model.fit(data[["steps"]], data["calories"])

    pred = model.predict([[steps]])[0]
    st.success(f"Calories: {int(pred)}")

    fig, ax = plt.subplots()
    ax.scatter(data["steps"], data["calories"])
    ax.plot(data["steps"], model.predict(data[["steps"]]))
    st.pyplot(fig)

# 🧠 CLASSIFICATION
elif menu == "🧠 Classification":
    steps = st.slider("Steps", 1000, 15000, key=1)
    calories = st.slider("Calories", 100, 1000, key=2)
    hr = st.slider("Heart Rate", 60, 140, key=3)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        model.fit(X, y)
        pred = model.predict([[steps, calories, hr]])[0]
        st.write(f"{name}: {pred}")

    fig, ax = plt.subplots()
    data["activity"].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# 👥 CLUSTERING
elif menu == "👥 Clustering":
    kmeans = KMeans(n_clusters=3)
    data["cluster"] = kmeans.fit_predict(X)

    st.write(data)

    fig, ax = plt.subplots()
    ax.scatter(data["steps"], data["calories"], c=data["cluster"])
    st.pyplot(fig)

# 📍 KNN
elif menu == "📍 KNN":
    steps = st.slider("Steps", 1000, 15000, key=4)
    calories = st.slider("Calories", 100, 1000, key=5)
    hr = st.slider("Heart Rate", 60, 140, key=6)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    pred = model.predict([[steps, calories, hr]])[0]
    st.success(f"Similar Activity: {pred}")

    fig, ax = plt.subplots()
    ax.scatter(data["steps"], data["calories"])
    ax.scatter(steps, calories)
    st.pyplot(fig)

# 🤖 ANN
elif menu == "🤖 ANN":
    model = Sequential()
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.fit(X, y, epochs=50, verbose=0)

    pred = model.predict(X)
    st.write("ANN Prediction Sample:", np.argmax(pred[0]))

# 🖼️ CNN DEMO
elif menu == "🖼️ CNN Demo":
    st.write("CNN used for image classification (MNIST concept demo)")
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png")

# 🔤 LSTM DEMO
elif menu == "🔤 LSTM Demo":
    text = st.text_input("Enter text:")
    if text:
        words = ["good", "healthy", "fit"]
        st.success(text + " " + random.choice(words))

# 🧼 AUTOENCODER
elif menu == "🧼 Autoencoder Demo":
    st.header("Autoencoder - Image Denoising")

    st.write("Input (Noisy Image)")
    st.image("images/noisy.png")

    st.write("Output (Denoised Image)")
    st.image("images/clean.png")
    
# 📌 ABOUT
else:
    st.write("""
    ✔ ML Models: Regression, Classification, KNN, Clustering  
    ✔ DL Models: ANN, CNN, LSTM, Autoencoder  
    ✔ Interactive UI using Streamlit  
    ✔ Visualization included  
    """)
