# 🚢 Titanic Survival Prediction Web App

This is a Streamlit-based web application that predicts Titanic passenger survival  
based on user input (e.g., age, sex, fare, etc.) and a trained ensemble model.

## 📦 Files

- `app.py`: Streamlit frontend UI for user input and prediction output  
- `predict.py`: Feature engineering logic to match model expectations  
- `model.pkl`: Trained stacking ensemble model (RandomForest + GBoost + XGBoost)  
- `requirements.txt`: Python dependencies for local or cloud deployment  

## ✨ Features

- **Inputs**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked  
- **Feature engineering**:  
  - Binning, one-hot encoding, family size, and several custom features  
- **Prediction**: Uses a stacking ensemble to determine survival  

## ✅ How to Run Locally

```bash
# 1. Activate virtual environment (Windows)
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
