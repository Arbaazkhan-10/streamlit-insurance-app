# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Step 1: Load dataset
df = pd.read_csv("insurance.csv")

# Step 2: Define features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Step 3: Handle categorical variables using OneHotEncoding
categorical_cols = ["sex", "smoker", "region"]
numeric_cols = ["age", "bmi", "children"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_cols)
], remainder='passthrough')

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 4: Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the model
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as insurance_model.pkl")
