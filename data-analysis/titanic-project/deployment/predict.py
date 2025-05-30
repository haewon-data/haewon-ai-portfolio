import joblib
import pandas as pd

# 저장된 모델 로드
model = joblib.load('model.pkl')  # deployment/ 폴더에 model.pkl 넣을 예정

# 예측 함수
def predict_survival(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction