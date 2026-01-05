import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title("📊 Linear Regression App (No Pickle)")

# Load data
df = pd.read_csv("Student_Performance.csv")

# Encode categorical
encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Train model
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

st.write("Enter feature values:")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    result = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Value: {result:.2f}")
