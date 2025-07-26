import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter the passenger details in the sidebar to predict survival.")

# Sidebar input
st.sidebar.header("Enter Passenger Details")
PassengerId = st.sidebar.number_input("Passenger ID", min_value=1, step=1)
Pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex", ["male", "female"])
Age = st.sidebar.slider("Age", 1, 80, 25)
Fare = st.sidebar.slider("Fare", 0, 500, 50)
Embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encoding
sex_map = {'male': 0, 'female': 1}
embarked_map = {'C': 0, 'Q': 1, 'S': 2}

# Convert input into DataFrame
input_dict = {
    'PassengerId': PassengerId,
    'Pclass': Pclass,
    'Sex': sex_map[Sex],
    'Age': Age,
    'Fare': Fare,
    'Embarked': embarked_map[Embarked]
}
input_df = pd.DataFrame([input_dict])

# Show Entered Data
st.subheader("Entered Data")
st.dataframe(input_df)

# Drop PassengerId before prediction
input_df = input_df.drop("PassengerId", axis=1)

# Prediction
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][prediction]

# Display Result
st.subheader("Prediction")
if prediction == 1:
    st.success(f"🟢 The passenger is likely to **Survive** ({probability*100:.2f}% probability)")
else:
    st.error(f"🔴 The passenger is likely to **Not Survive** ({probability*100:.2f}% probability)")
