import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def generate_synthetic_data(
    n_samples: int=50_000,
    rng_seed: int=42
) -> pd.DataFrame:
    """
    Генерирует датасет с вещественными признаками и дисбалансом целевого класса.

    Args:
        n_samples: int
            Общее количество строк датасета. По умолчанию 50_000.
        seed: int
            Начальное значение (seed) для генератора случайных чисел
            `np.random.default_rng`. По умолчанию 42.

    Returns:
        pd.DataFrame: Сгенерированный датасет со столбцами 'feature_1',
            'feature_2', 'feature_3' и 'target'.

    Raises:
        ValueError: Если `n_samples` меньше или равен нулю.

    Notes:
        Целевая переменная 'target' содержит ровно 5% единиц и 95% нулей,
        что имитирует задачу классификации в условиях сильного дисбаланса.
    """
    if n_samples <= 0:
        raise ValueError("Параметр n_samples должен быть > 0.")
    
    rng = np.random.default_rng(seed=rng_seed)
    
    # Вычисляем точное количество единиц (5% от n_samples) и нулей
    n_ones: int = int(n_samples * 0.05)
    n_zeros: int = n_samples - n_ones
    
    # Создаем точный бинарный вектор целевой переменной и перемешиваем его
    target: np.ndarray = np.array([0] * n_zeros + [1] * n_ones, dtype=np.int64)
    rng.shuffle(target)
    
    # Генерация вещественных признаков
    feature_1: np.ndarray = rng.standard_normal(n_samples)
    feature_2: np.ndarray = rng.uniform(-5.0, 5.0, size=n_samples)
    feature_3: np.ndarray = rng.normal(loc=10.0, scale=2.0, size=n_samples)
    
    df = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "target": target,
    })
    
    return df

def split_data(
    df: pd.DataFrame,
    target_col: str="target",
    test_size: float=0.2,
    random_state: int=42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет датасет на train/test с фиксацией random_state и стратификацией.

    Args:
        df: pd.DataFrame
            Исходный датасет для разделения.
        target_col: str
            Название столбца с целевой переменной. По умолчанию "target".
        test_size: float
            Доля тестовой выборки от общего объема данных (от 0.0 до 1.0).
            По умолчанию 0.2.
        random_state: int
            Сид для воспроизводимости результатов разделения.
            По умолчанию 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Кортеж из четырех элементов:
            - X_train (pd.DataFrame): Признаки обучающей выборки.
            - X_test (pd.DataFrame): Признаки тестовой выборки.
            - y_train (pd.Series): Целевая переменная обучающей выборки.
            - y_test (pd.Series): Целевая переменная тестовой выборки.

    Raises:
        KeyError: Если столбец `target_col` отсутствует в переданном DataFrame.
        ValueError: Если `test_size` выходить за пределы допустимого интервала (0.0, 1.0).

    Notes:
        Использование `stratify=y` гарантирует сохранение пропорций классов
        (95% нулей и 5% единиц) как в обучающей, так и в тестовой выборках.
    """
    if target_col not in df.columns:
        raise KeyError(f"Столбец '{target_col}' отсутствует в переданном DataFrame.")
    
    if not (0.0 < test_size < 1.0):
        raise ValueError("Параметр test_size должен находиться в диапазоне (0.0, 1.0).")
    
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def main() -> None:
    """
    Запускает полный пайплайн:
        - генерацию данных
        - разделение
        - масштабирование
        - обучение модели
        - получение предсказаний
    """
    
    # Генерация данных
    raw_df = generate_synthetic_data(n_samples=50_000, rng_seed=42)
    print(f"Сгенерированный DataFrame:\n{raw_df}\n")
    
    class_proportions = raw_df["target"].value_counts(normalize=True)
    print(f"Доли классов в исходном DataFrame:\n{class_proportions}\n")
    
    # Разделение на train и test (80/20)
    X_train, X_test, y_train, y_test = split_data(
        raw_df,
        target_col="target",
        test_size=0.2,
        random_state=42
    )
    
    # Инициализация и масштабирование
    scaler = StandardScaler()
    
    # Обучаем scaler только на train, трансформируем train и test
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)
    
    # Обучение модели
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Получение вероятностей для тестовой выборки
    probabilities: np.ndarray = model.predict_proba(X_test_scaled)
    positive_class_probs: np.ndarray = probabilities[:, 1]
    
    print(f"model.predict_proba(X_test_scaled):\n{probabilities}\n")
    
    # Вывод первых 10 значений
    print("Первые 10 вероятностей для положительного класса (1):")
    for i, prob in enumerate(positive_class_probs[:10], 1):
        print(f"Объект {i}: {prob:.6f}")

if __name__ == "__main__":
    main()