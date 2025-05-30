import streamlit as st
from predict import predict_survival

st.title("🚢 Titanic 생존 예측기")
st.write("탑승 정보를 입력하면 생존 여부를 예측해줍니다.")

# 사용자 입력 받기
pclass = st.selectbox("Pclass (객실 등급)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("형제/배우자 수 (SibSp)", 0, 10, 0)
parch = st.number_input("부모/자녀 수 (Parch)", 0, 10, 0)
fare = st.number_input("탑승 요금 (Fare)", 0.0, 600.0, 30.0)
embarked = st.selectbox("탑승 항구 (Embarked)", ["C", "Q", "S"])

if st.button("예측하기"):
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
    st.success(f"예측 결과: {'생존' if result == 1 else '사망'}")