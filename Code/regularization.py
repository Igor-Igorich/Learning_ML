import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 15,
    n_informative: int = 5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Генерирует датасет для задачи бинарной классификации со сгенерированным шумом.

    Args:
        n_samples (int, optional): Количество наблюдений (строк). По умолчанию 10000.
        n_features (int, optional): Общее количество признаков. По умолчанию 15.
        n_informative (int, optional): Количество сильных информативных признаков. По умолчанию 5.
        seed (int, optional): Сид для воспроизводимого генератора случайных чисел. По умолчанию 42.

    Raises:
        ValueError: Если количество информативных признаков превышает общее 
            число признаков или если n_samples <= 0.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Кортеж из матрицы признаков X (DataFrame) 
            и вектора целевой переменной y (Series).
    """


    if n_informative > n_features:
        raise ValueError("Число информативных признаков не может превышать общее число признаков.")
    if n_samples <= 0:
        raise ValueError("Количество образцов должно быть положительным числом.")
    
    rng = np.random.default_rng(seed=seed)
    
    # 1. Генерация матрицы признаков X ~ N(0, 1)
    x_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # 2. Задание истинных весов: первые n_informative весовые коэффициенты сильные, остальные — ровно 0
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = rng.uniform(1.5, 3.0, size=n_informative) * rng.choice([-1, 1], size=n_informative)

    # 3. Вычисление логитов z = Xw + noise
    logits = x_matrix @ true_weights + rng.normal(loc=0.0, scale=0.5, size=n_samples)
    probabilities = 1.0 / (1.0 + np.exp(-logits)) # сигмоидная функция

    # 4. Бинарный целевой класс по распределению Бернулли
    y_vector = rng.binomial(n=1, p=probabilities)
    
    feature_names = [
        f"feature_{i+1:02d}_(signal)" if i < n_informative else f"feature_{i+1:02d}_(noise)"
        for i in range(n_features)
    ]
    
    x_df = pd.DataFrame(x_matrix, columns=feature_names)
    y_series = pd.Series(y_vector, name="target")
    
    return x_df, y_series

def run_regularization_experiment(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    c_values: List[float]
) -> None:
    """Обучает модели LogisticRegression с L1 и L2 регуляризацией для разных C и выводит полученные веса.

    Args:
        x_df (pd.DataFrame): Матрица признаков.
        y_series (pd.Series): Вектор целевой переменной.
        c_values (List[float]): Список значений параметров C (обратная сила регуляризации).
    
    Returns:
        None

    Raises:
        ValueError: Если список c_values пуст.

    Notes:
        Перед обучением применяется StandardScaler, так как регуляризация 
        чувствительна к масштабу признаков.
        L1 использует solver='liblinear', L2 использует solver='lbfgs'.
    """
    
    if not c_values:
        raise ValueError("Список c_values не должен быть пустым.")
    
    # Стандартизация признаков (Mean=0, Std=1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)
    
    logger.info("Запуск эксперимента с L1 (liblinear) и L2 (lbfgs) регуляризацией...")
    
    for c_param in c_values:
        logger.info("=" * 70)
        logger.info(f"Сила регуляризации C = {c_param} (лямбда = {1/c_param:.3f})")
        logger.info("=" * 70)
        
        model_l1 = LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=c_param,
            random_state=42
        )
        model_l1.fit(x_scaled, y_series)
        weights_l1 = model_l1.coef_[0]
        
        model_l2 = LogisticRegression(
            l1_ratio=0.0,
            solver='lbfgs',
            C=c_param,
            random_state=42
        )
        model_l2.fit(x_scaled, y_series)
        weights_l2 = model_l2.coef_[0]
        
        # Подсчет обнуленных весов
        zeros_l1 = np.sum(weights_l1 == 0)
        zeros_l2 = np.sum(weights_l2 == 0)
        
        comparison_df = pd.DataFrame({
            "Feature Name": x_df.columns,
            "L1 Weight (liblinear)": np.round(weights_l1, 4),
            "L2 Weight (lbfgs)": np.round(weights_l2, 4)
        })
        
        logger.info(f"Строгих нулей в L1: {zeros_l1} из {len(weights_l1)}")
        logger.info(f"Строгих нулей в L2: {zeros_l2} из {len(weights_l2)}")
        logger.info("\nТаблица полученных весов:\n" + comparison_df.to_string(index=False))
        

def main() -> None:
    """Главная управляющая функция скрипта."""
    
    logger.info("Старт программы.")

    x_data, y_data = generate_synthetic_data(
        n_samples=10000,
        n_features=15,
        n_informative=5,
        seed=42
    )
    
    c_list = [0.001, 0.1, 1.0, 100.0]

    run_regularization_experiment(x_data, y_data, c_list)

    logger.info("Эксперимент успешно завершен.")

if __name__ == "__main__":
    main()