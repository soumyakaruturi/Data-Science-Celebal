import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model
model = joblib.load("model.pkl")
iris = load_iris()

st.set_page_config(page_title="Iris Flower Predictor", layout="centered")
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter measurements and predict the Iris species!")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_class = iris.target_names[prediction]

st.subheader("🔍 Prediction")
st.success(f"The predicted Iris species is **{predicted_class}**")

# Display probabilities
probs = model.predict_proba(input_data)
st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame(probs, columns=iris.target_names)
st.dataframe(prob_df.style.highlight_max(axis=1, color="lightgreen"))

# Visual explanation
st.subheader("📈 Input Compared to Dataset")
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

fig, ax = plt.subplots()
sns.scatterplot(
    data=iris_df, x='petal length (cm)', y='petal width (cm)',
    hue='species', palette='Set1', alpha=0.6, s=60, ax=ax
)
plt.scatter(input_data[0][2], input_data[0][3], color='black', s=100, label="Your Input", marker='X')
plt.legend()
st.pyplot(fig)

st.caption("Made with ❤️ using Streamlit")
