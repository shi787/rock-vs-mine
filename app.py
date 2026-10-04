import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Rock vs Mine", page_icon="🌊")
st.title("Sonar: Rock vs Mine Predictor")

DATA_PATH = "Copy of sonar data.csv"  # keep this file in the same folder as app.py


@st.cache_data
def load_data():
    # The classic sonar dataset has no header row: 60 features + label (R or M)
    return pd.read_csv(DATA_PATH, header=None)


@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop(columns=60)
    y = df[60]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=1
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


df = load_data()
model, acc = train_model()
st.caption(f"Model: Logistic Regression | Test accuracy: {acc:.2%}")

st.subheader("Enter sonar readings")
st.write("Paste 60 comma-separated values, or load a sample row from the dataset.")

if "text" not in st.session_state:
    st.session_state.text = ""

if st.button("Load random sample row"):
    row = df.sample(1).iloc[0]
    st.session_state.text = ",".join(str(v) for v in row[:60])
    st.session_state.actual = row[60]

text = st.text_area("60 values", key="text", height=150)

if st.button("Predict"):
    try:
        values = [float(v) for v in text.split(",") if v.strip()]
        if len(values) != 60:
            st.error(f"Need exactly 60 values, got {len(values)}.")
        else:
            x = np.array(values).reshape(1, -1)
            pred = model.predict(x)[0]
            if pred == "M":
                st.error("The object is a **Mine** 💣")
            else:
                st.success("The object is a **Rock** 🪨")
            if "actual" in st.session_state:
                st.caption(f"(Actual label of the loaded sample: {st.session_state.actual})")
    except ValueError:
        st.error("Please enter only numbers separated by commas.")