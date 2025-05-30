import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Feature names used during model training (must match exactly)
expected_columns = [
    'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E',
    'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U',
    'Pclass_1', 'Pclass_2', 'Pclass_3',
    'FamilySize', 'IsAlone', 'LowFare', 'AgeBin',
    'ModerateFamily', 'IsCherbourg', 'FemaleFirstSecondClass',
    'IsChildOrElderly', 'LowFare_3rdClass',
    'Sex_male', 'Sex_female', 'Age', 'Parch', 'SibSp'
]

# Prediction function
def predict_survival(input_data):
    df = pd.DataFrame([input_data])

    # One-hot encode 'Sex'
    df['Sex_male'] = (df['Sex'] == 'male').astype(int)
    df['Sex_female'] = (df['Sex'] == 'female').astype(int)
    df.drop(columns='Sex', inplace=True)

    # One-hot encode 'Embarked'
    df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
    df.drop(columns='Embarked', inplace=True)

    # One-hot encode 'Pclass'
    df['Pclass_1'] = (df['Pclass'] == 1).astype(int)
    df['Pclass_2'] = (df['Pclass'] == 2).astype(int)
    df['Pclass_3'] = (df['Pclass'] == 3).astype(int)
    df.drop(columns='Pclass', inplace=True)

    # Family-related features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['ModerateFamily'] = df['FamilySize'].between(2, 4).astype(int)

    # Fare-related features
    df['LowFare'] = (df['Fare'] < 10).astype(int)
    df['LowFare_3rdClass'] = ((df['Fare'] < 10) & (df['Pclass_3'] == 1)).astype(int)

    # Age-related features
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 60, 100], labels=[0, 1, 2]).astype(int)
    df['IsChildOrElderly'] = ((df['Age'] <= 12) | (df['Age'] >= 60)).astype(int)

    # Additional features
    df['IsCherbourg'] = (df['Embarked_C'] == 1).astype(int)
    df['FemaleFirstSecondClass'] = ((df['Sex_female'] == 1) & (df['Pclass_1'] + df['Pclass_2'] >= 1)).astype(int)

    # Cabin-related features: Assume unknown cabin (Cabin_U = 1, others = 0)
    for c in ['A','B','C','D','E','F','G','T']:
        df[f'Cabin_{c}'] = 0
    df['Cabin_U'] = 1

    # Reorder columns to match training data
    df = df.reindex(columns=expected_columns, fill_value=0)

    print("Prediction input shape:", df.shape)
    print("Prediction input columns:", df.columns.tolist())
    print("Prediction input columns:", df.columns.tolist())
    print("Model output:", model.predict(df))


    # Predict
    prediction = model.predict(df)[0]
    return "Survived" if prediction == 1 else "Did not survive"
