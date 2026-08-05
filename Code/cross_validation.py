import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# 1. Генерация синтетических данных (Истинная функция: y = x * sin(x) + шум)
np.random.seed(42)
n_samples = 200

# Генерируем признак X в виде вектора-столбца (матрица объект-признак)
X = np.random.uniform(0, 10, size=(n_samples, 1))

# Генерируем целевую переменную y с добавлением нормального шума (epsilon)
noise = np.random.normal(0, 1.5, size=(n_samples, 1))
y = X * np.sin(X) + noise
y = y.ravel() # Переводим в одномерный массив для sklearn

# Упакуем в Pandas DataFrame для удобства первичного анализа
df = pd.DataFrame({'Feature_X': X.ravel(), 'Target_Y': y})
print("Пример сгенерированных данных:")
print(df.head(), "\n")

# 2. Настройка схемы валидации (K-Fold)
# shuffle=True обязателен, чтобы избежать влияния исходной сортировки данных
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Определение моделей разной сложности для демонстрации Bias-Variance Tradeoff
# Используем Pipeline для изоляции фолдов и предотвращения утечки данных (Data Leakage).
# Трансформации (StandardScaler, PolynomialFeatures) будут обучаться ТОЛЬКО на X_train каждого фолда.

pipelines = {
    # Underfitting (Высокий Bias): Простая линейная модель не способна описать нелинейность x*sin(x)
    "Underfitting (Degree 1)": Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=1, include_bias=False)),
        ("model", Ridge(alpha=1.0))
    ]),
    
    # Optimal (Баланс Bias и Variance): Полином 5-й степени адекватно приближает кривую
    "Optimal (Degree 5)": Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("model", Ridge(alpha=1.0))
    ]),
    
    # Overfitting (Высокий Variance): Полином 15-й степени вызубрит шум (epsilon) обучающей выборки
    "Overfitting (Degree 15)": Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=15, include_bias=False)),
        ("model", Ridge(alpha=1e-4)) # Уменьшаем регуляризацию, чтобы позволить модели переобучиться
    ])
}

# 4. Проведение кросс-валидации и сбор метрик
results = []

for name, pipeline in pipelines.items():
    # Используем cross_validate для получения метрик и на обучении, и на тесте (валидации)
    # scoring='neg_mean_squared_error', так как sklearn максимизирует метрики
    cv_scores = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=kf,
        scoring='neg_mean_squared_error',
        return_train_score=True # Включаем расчет ошибки на обучающих фолдах!
    )
    
    # Переводим negative MSE в обычный MSE
    train_mse = -cv_scores['train_score'].mean()
    val_mse = -cv_scores['test_score'].mean()
    
    results.append({
        "Модель": name,
        "MSE (Train)": round(train_mse, 3),
        "MSE (Validation)": round(val_mse, 3),
        "Разница (Val - Train)": round(val_mse - train_mse, 3)
    })

# 5. Анализ результатов с помощью Pandas
results_df = pd.DataFrame(results)
results_df.set_index("Модель", inplace=True)

print("Результаты 5-Fold кросс-валидации:")
print("-" * 60)
print(results_df)
print("-" * 60)