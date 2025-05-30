import streamlit as st
from predict import predict_survival

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger information below to predict survival outcome.")

# Collect user input
pclass = st.selectbox("Pclass (Passenger Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Make prediction when button is clicked
if st.button("Predict"):
    input_data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }
    result = predict_survival(input_data)
    if result == 'Survived':
        st.success("Prediction: Survived")
    elif result == 'Did not survive':
        st.error("Prediction: Did not survive")
    else:
        st.warning(f"Unexpected prediction result: {result}")