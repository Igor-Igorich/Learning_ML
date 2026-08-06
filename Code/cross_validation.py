'''
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
'''

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------------------------------------------------------------------
# 1. Генерация синтетического датасета с шумом
# ------------------------------------------------------------------------------
# flip_y=0.15 зашумляет 15% меток классов (случайно меняет их на противоположные).
# n_informative=6, n_redundant=2 добавляют реалистичную структуру признаков.
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    flip_y=0.15,
    random_state=42
)

print(f'X:\n{X}')
print(f'y:\n{y}')

# Оборачиваем моделирование в Pipeline для правильного масштабирования признаков без Data Leakage
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

# ------------------------------------------------------------------------------
# 2. Одиночное разбиение (Holdout validation: Train / Test)
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение и оценка
pipeline.fit(X_train, y_train)
single_split_acc = pipeline.score(X_test, y_test)

print("=" * 65)
print("1. ОДИНОЧНОЕ РАЗБИЕНИЕ (Train/Test Split 80/20)")
print(f"   Accuracy на тестовой выборке: {single_split_acc:.4f}")
print("=" * 65)

# ------------------------------------------------------------------------------
# 3. Кросс-валидация (5-Fold Cross-Validation)
# ------------------------------------------------------------------------------
# Для классификации по умолчанию в sklearn используется StratifiedKFold
cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    estimator=pipeline,
    X=X,
    y=y,
    cv=cv_scheme,
    scoring='accuracy'
)

print("\n2. КРОСС-ВАЛИДАЦИЯ (5-Fold CV)")
print(f"   Массив оценок по фолдам:  {np.round(cv_scores, 4)}")
print(f"   Среднее значение (Mean):  {cv_scores.mean():.4f}")
print(f"   Стандартное отклонение:   {cv_scores.std():.4f}")
print("=" * 65)

# ------------------------------------------------------------------------------
# 4. АНАЛИТИЧЕСКИЙ КОММЕНТАРИЙ
# ------------------------------------------------------------------------------
"""
АНАЛИЗ НАДЕЖНОСТИ ПОДХОДОВ:

1. Одиночное разбиение (Holdout):
   - Оценка метрики (Accuracy = {:.4f}) зависит от точного случайного разделения
     данных на train/test. При изменении random_state точечная оценка может сместиться.
   - Подход обучается только на 80% данных, теряя часть информации для обучения.

2. Кросс-валидация (5-Fold CV):
   - Дает робастную оценку математического ожидания ошибки ({:.4f}) и дисперсию
     модели (std = {:.4f}) на различных подвыборках.
   - Каждый объект выборки успевает побывать в валидационном фолде ровно 1 раз.
   - Стандартное отклонение ({:.4f}) показывает, насколько стабильно ведет себя
     модель: низкое std подтверждает, что логистическая регрессия не переобучается
     и сохраняет ровное качество независимо от разбиения.

ВЫВОД:
Для зашумленных данных кросс-валидация значительно надежнее, так как позволяет
отделить реальный сигнал модели от случайного попадания "удачных" или "неудачных"
объектов в единственный валидационный сет.
""".format(single_split_acc, cv_scores.mean(), cv_scores.std(), cv_scores.std())