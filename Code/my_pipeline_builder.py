import logging
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

_cached_pipeline: Optional[Pipeline] = None


def generate_synthetic_data(
    n_samples: int = 5000,
    missing_rate: float = 0.10,
    cols_with_missing: Optional[List[str]] = None,
    noise_level: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Генерирует датасет, математически согласованный с линейными допущениями LogisticRegression."""
    rng = np.random.default_rng(random_state)

    num_cols = ["num_1", "num_2", "num_3", "num_4"]
    num_data = rng.normal(size=(n_samples, 4))
    X = pd.DataFrame(num_data, columns=num_cols)

    cat_definitions = {
        "cat_1": ["low", "medium", "high"],
        "cat_2": ["red", "blue", "green", "yellow"],
        "cat_3": ["city_A", "city_B"],
        "cat_4": ["tier_1", "tier_2", "tier_3"],
    }

    for col_name, cat in cat_definitions.items():
        X[col_name] = rng.choice(cat, size=n_samples)

    beta_num = np.array([0.8, -1.2, 0.5, 0.0])

    cat_weights = {
        "cat_1": {"low": -0.6, "medium": 0.0, "high": 1.2},
        "cat_2": {"red": 0.4, "blue": -0.4, "green": 0.0, "yellow": 0.1},
        "cat_3": {"city_A": -0.5, "city_B": 0.5},
        "cat_4": {"tier_1": 0.7, "tier_2": 0.0, "tier_3": -0.7},
    }

    z = X[num_cols].values @ beta_num

    for col_name, weight_map in cat_weights.items():
        z += X[col_name].map(weight_map).values

    z += rng.normal(scale=noise_level, size=n_samples)
    probabilities = 1.0 / (1.0 + np.exp(-z))

    y = rng.binomial(n=1, p=probabilities)
    y = pd.Series(y, name="target")

    if cols_with_missing is None:
        cols_with_missing = ["num_1", "num_2", "cat_1"]

    for col in cols_with_missing:
        mask = rng.random(n_samples) < missing_rate
        X.loc[mask, col] = np.nan

    return X, y


def predict_proba_lazy(
    X_new: pd.DataFrame, model_path: str = "ml_pipeline.joblib"
) -> np.ndarray:
    """Выполняет ленивую загрузку модели при первом вызове

    и возвращает вероятности классификации для X_new.
    """
    global _cached_pipeline

    if _cached_pipeline is None:
        logger.info("Загрузка модели из диска (%s)...", model_path)
        _cached_pipeline = joblib.load(model_path)
    else:
        logger.info("Использование модели из кэша OZU.")

    return _cached_pipeline.predict_proba(X_new)


if __name__ == "__main__":
    rnd_seed = 42

    logger.info("Генерация синтетических данных...")
    X, y = generate_synthetic_data(
        n_samples=5000,
        missing_rate=0.10,
        cols_with_missing=["num_1", "num_2", "cat_1"],
        noise_level=0.5,
        random_state=rnd_seed,
    )

    num_cols = ["num_1", "num_2", "num_3", "num_4"]
    cat_cols = ["cat_1", "cat_2", "cat_3", "cat_4"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rnd_seed, stratify=y
    )

    num_pipeline = Pipeline(
        [
            ("num_imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=rnd_seed)),
        ]
    )

    logger.info("Обучение Pipeline...")
    full_pipeline.fit(X_train, y_train)

    test_probas = full_pipeline.predict_proba(X_test)
    logger.info(
        "Получены вероятности для %d тестовых объектов.", len(test_probas)
    )

    model_filename = "ml_pipeline.joblib"
    joblib.dump(full_pipeline, model_filename)
    logger.info("Модель успешно сохранена в '%s'", model_filename)

    res_first_call = predict_proba_lazy(X_test.iloc[:3], model_filename)
    res_second_call = predict_proba_lazy(X_test.iloc[:3], model_filename)
