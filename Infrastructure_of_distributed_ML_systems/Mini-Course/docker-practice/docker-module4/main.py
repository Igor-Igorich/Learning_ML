import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.linear_model import LinearRegression

app = FastAPI(title="ML Model API")

model = LinearRegression()
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2.1, 4.0, 6.1, 7.8, 10.2])
model.fit(X_train, y_train)


@app.get("/")
def root():
    return {"message": "ML API is running", "model": "LinearRegression"}


@app.get("/predict")
def predict(x: float):
    prediction = model.predict(np.array([[x]]))[0]
    return {
        "input": x,
        "prediction": round(float(prediction), 4),
        "model_type": "scikit-learn",
    }


@app.get("/stats")
def stats():
    # Демонстрация работы pandas
    df = pd.DataFrame(
        {"feature": [1, 2, 3, 4, 5], "target": [2.1, 4.0, 6.1, 7.8, 10.2]}
    )
    return {
        "mean": float(df["target"].mean()),
        "std": float(df["target"].std()),
    }
