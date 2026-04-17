import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ---------------- PAGE ----------------
st.set_page_config(page_title="Fitness AI System", layout="wide")


st.title("🏃 Smart Fitness Activity Prediction System")


# ---------------- DATA ----------------
data = pd.DataFrame({
    "steps": [2000, 5000, 8000, 12000, 3000, 7000, 9000, 11000],
    "calories": [150, 300, 500, 800, 200, 450, 600, 750],
    "heart_rate": [70, 85, 95, 110, 75, 90, 100, 105],
    "activity": [0, 1, 2, 2, 0, 1, 2, 2]
})


X = data[["steps", "calories", "heart_rate"]]
y = data["activity"]


menu = st.sidebar.selectbox("Menu", [
    "📌 Problem Statement",
    "📊 Data Analysis",
    "📈 Regression",
    "🧠 Classification",
    "📍 KNN",
    "👥 Clustering",
    "🤖 ANN",
    "📊 Performance",
    "📌 Conclusion"
])


# ---------------- PROBLEM ----------------
if menu == "📌 Problem Statement":
    st.subheader("Problem Statement")


    st.write("""
    In today's digital world, fitness apps and wearable devices generate large amounts of user data such as steps, calories burned, and heart rate.
   
    However, this data is often not analyzed effectively, and users fail to gain meaningful insights about their health and activity levels.
   
    This project aims to build an intelligent system that can:
    - Analyze fitness data
    - Identify patterns
    - Predict user activity levels using Machine Learning and Deep Learning models
    """)


    st.subheader("Why This Problem?")
    st.write("""
    - Increasing use of fitness trackers
    - Lack of intelligent analysis tools
    - Need for personalized health insights
    """)


    st.subheader("Observations")
    st.write("""
    - Higher steps generally lead to higher calorie burn
    - Increased heart rate indicates higher physical activity
    - Users with similar patterns can be grouped together
    """)


    st.subheader("What Makes This Project Different?")
    st.write("""
    - Combines ML + DL in one system
    - Uses multiple algorithms for comparison
    - Provides visual insights using graphs
    - Interactive prediction system
    """)


# ---------------- DATA ANALYSIS ----------------
elif menu == "📊 Data Analysis":
    st.subheader("Dataset")
    st.dataframe(data)


    col1, col2 = st.columns(2)


    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.scatter(data["steps"], data["calories"])
        ax.set_title("Steps vs Calories")
        st.pyplot(fig)


    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(data["heart_rate"], bins=5)
        ax.set_title("Heart Rate Distribution")
        st.pyplot(fig)


# ---------------- REGRESSION ----------------
elif menu == "📈 Regression":
    st.subheader("Linear Regression")


    model = LinearRegression()
    model.fit(data[["steps"]], data["calories"])


    steps = st.slider("Steps", 1000, 15000, 8000)
    pred = model.predict([[steps]])[0]


    st.success(f"Predicted Calories: {int(pred)}")


    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(data["steps"], data["calories"])
    ax.set_title("Regression Line")
    st.pyplot(fig)


# ---------------- CLASSIFICATION ----------------
elif menu == "🧠 Classification":
    st.subheader("Classification Models")


    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }


    acc_list = []


    for name, model in models.items():
        model.fit(X, y)
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        acc_list.append(acc)


        st.write(f"{name} Accuracy: {round(acc,2)}")


    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(list(models.keys()), acc_list)
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)


# ---------------- KNN ----------------
elif menu == "📍 KNN":
    st.subheader("KNN Prediction")


    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)


    s = st.number_input("Steps", value=6000)
    c = st.number_input("Calories", value=400)
    hr = st.number_input("Heart Rate", value=85)


    pred = model.predict([[s, c, hr]])
    st.success(f"Predicted Activity Level: {pred[0]}")


# ---------------- CLUSTERING ----------------
elif menu == "👥 Clustering":
    st.subheader("K-Means Clustering")


    kmeans = KMeans(n_clusters=3, n_init=10)
    data["cluster"] = kmeans.fit_predict(X)


    st.dataframe(data)


    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(data["steps"], data["calories"], c=data["cluster"])
    ax.set_title("Clusters")
    st.pyplot(fig)


# ---------------- ANN ----------------
elif menu == "🤖 ANN":
    st.subheader("Artificial Neural Network")


    model = Sequential()
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(3, activation='softmax'))


    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=50, verbose=0)


    loss, acc = model.evaluate(X, y, verbose=0)
    st.success(f"ANN Accuracy: {round(acc,2)}")


# ---------------- PERFORMANCE ----------------
elif menu == "📊 Performance":
    st.subheader("Performance Evaluation")


    model = RandomForestClassifier()
    model.fit(X, y)


    pred = model.predict(X)


    st.write("Accuracy:", accuracy_score(y, pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y, pred))


# ---------------- CONCLUSION ----------------
elif menu == "📌 Conclusion":
    st.subheader("Conclusion")


    st.write("""
    This project demonstrates how Machine Learning and Deep Learning techniques can be applied to fitness data analysis.


    Key Results:
    - Random Forest achieved high accuracy
    - ANN provided strong predictive capability
    - Clustering revealed user activity patterns


    This system can be further extended into real-time health monitoring applications.
    """)



