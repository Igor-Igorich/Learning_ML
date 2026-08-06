import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_imbalanced_data(
    n_samples: int=5000,
    n_features: int=10,
    weights: List[float]=[0.95, 0.05],
    seed: int=42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Генерирует синтетический несбалансированный датасет для задачи бинарной классификации.

    Args:
        n_samples (int, optional): Общее количество генерируемых объектов. По умолчанию 5000.
        n_features (int, optional): Кол-во признаков. По умолчанию 10.
        weights (List[float], optional): Соотношение классов. По умолчанию [0.95, 0.05].
        seed (int, optional): Сид для генератора случайных чисел. По умолчанию 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - X (pd.DataFrame): Таблица признаков объектов.
            - y (pd.Series): Вектор целевых меток классов (0 и 1).

    Raises:
        ValueError: Если n_samples <= 0 или сумма элементов массива weights не равна 1.0.
    """
    
    if n_samples <= 0:
        raise ValueError("Параметр n_samples должен быть строго больше 0.")
    if not np.isclose(sum(weights), 1.0): # np.isclose — сравнение чисел с плавающей точкой на равенство с учетом допуска
        raise ValueError("Сумма элементов массива weights должна быть равна 1.0.")
    
    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        weights=weights,
        random_state=seed
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y_raw, name="target")
    
    return X, y


def calculate_manual_metrics(
    tn: int,
    fp: int,
    fn: int,
    tp: int
) -> Dict[str, float]:
    """
    Вычисляет базовые метрики классификации вручную на основе матрицы ошибок.

    Args:
        tn (int): Количество истинно отрицательных срабатываний (True Negatives).
        fp (int): Количество ложно положительных срабатываний (False Positives).
        fn (int): Количество ложно отрицательных срабатываний (False Negatives).
        tp (int): Количество истинно положительных срабатываний (True Positives).

    Returns:
        Dict[str, float]: Словарь со значениями расчитанных метрик:
            - 'accuracy': Доля правильных ответов.
            - 'precision': Точность.
            - 'recall': Полнота.
            - 'f1_score': F1-мера (гармоническое среднее precision и recall).

    Raises:
        ZeroDivisionError: Если общее количество элементов равна 0.

    Notes:
        Математические формулы расчета:
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1-Score = 2 * TP / (2 * TP + FP + FN)
    """
    
    total = tp + tn + fp + fn
    if total == 0:
        raise ZeroDivisionError("Общая сумма элементов матрицы ошибок равен 0.")
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Защита от деления на ноль
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Защита от деления на ноль
    f1 = 2 * tp / (2 * tp + fp + fn)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def main() -> None:
    """Точка входа в программу."""
    
    rnd_seed = 42
    
    logger.info("Генерация синтетического несбалансированного датасета...")
    X, y = generate_imbalanced_data(
        n_samples=5000,
        n_features=10,
        weights=[0.95, 0.05],
        seed=rnd_seed
    )

    class_distribution = y.value_counts(normalize=True) * 100
    logger.info(
        f"Распределение классов (%):\n{class_distribution.to_string()}"
    )
    
    logger.info("Разделение выборки на обучающую и тестовую (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=rnd_seed,
        stratify=y
    )
    
    logger.info("Обучение модели LogisticRegression...")
    model = LogisticRegression(random_state=rnd_seed, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    logger.info("Вычисление метрик с помощью библиотечных функций scikit-learn...")
    cm_sklearn = confusion_matrix(y_test, y_pred)
    acc_sklearn = accuracy_score(y_test, y_pred)
    prec_sklearn = precision_score(y_test, y_pred, zero_division=0)
    rec_sklearn = recall_score(y_test, y_pred, zero_division=0)
    f1_sklearn = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"\n--- Библиотечные метрики (Scikit-Learn) ---")
    logger.info(f"Confusion Matrix:\n{cm_sklearn}")
    logger.info(f"Accuracy:  {acc_sklearn:.6f}")
    logger.info(f"Precision: {prec_sklearn:.6f}")
    logger.info(f"Recall:    {rec_sklearn:.6f}")
    logger.info(f"F1-score:  {f1_sklearn:.6f}")
    
    logger.info("\nВычисление метрик кастомной функцией...")
    tn, fp, fn, tp = cm_sklearn.ravel()
    logger.info(f"Компоненты матрицы ошибок: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    manual_metrics = calculate_manual_metrics(tn=tn, fp=fp, fn=fn, tp=tp)

    logger.info(f"\n--- Результаты ручного расчета ---")
    logger.info(f"Accuracy:  {manual_metrics['accuracy']:.6f}")
    logger.info(f"Precision: {manual_metrics['precision']:.6f}")
    logger.info(f"Recall:    {manual_metrics['recall']:.6f}")
    logger.info(f"F1-score:  {manual_metrics['f1_score']:.6f}")
    
    logger.info("\nСверка результатов (Sklearn vs Manual):")
    is_acc_match = np.isclose(acc_sklearn, manual_metrics["accuracy"])
    is_prec_match = np.isclose(prec_sklearn, manual_metrics["precision"])
    is_rec_match = np.isclose(rec_sklearn, manual_metrics["recall"])
    is_f1_match = np.isclose(f1_sklearn, manual_metrics["f1_score"])

    if all([is_acc_match, is_prec_match, is_rec_match, is_f1_match]):
        logger.info("УСПЕХ: Все ручные метрики полностью совпадают с библиотечными!")
    else:
        logger.error("ОШИБКА: Обнаружены расхождения в расчетах!")

if __name__ == "__main__":
    main()