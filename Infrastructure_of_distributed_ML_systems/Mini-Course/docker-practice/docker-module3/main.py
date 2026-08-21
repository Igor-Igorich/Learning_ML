from fastapi import FastAPI

app = FastAPI(title="ML Prediction API")


@app.get("/")
def read_root():
    return {"message": "Hello from Docker!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/pregict")
def predict(x: float = 1.0, y: float = 2.0):
    # Условная ML-модель: простое предсказание
    result = x * 2.5 + y * 1.8 + 0.5
    return {
        "input": {"x": x, "y": y},
        "prediction": result,
        "model_version": "1.0.2",  # внесли изменения 1.0.0 --> 1.0.1 --> 1.0.2
    }
