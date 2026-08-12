import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VIFSelector(BaseEstimator, TransformerMixin):
    """
    Трансформер scikit-learn для итеративной фильтрации признаков по VIF.
    На каждом шаге удаляет один признак с VIF > threshold.
    """

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold
        self.dropped_cols_ = []
        self.selected_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Матрица X должна иметь тип данных pd.DataFrame")

        X_curr = X.copy()

        while True:
            # Statsmodels требует константу для корректной оценки R^2 во вспомогательных регрессиях
            if X_curr.shape[1] <= 1:
                break

            X_with_const = sm.add_constant(X_curr)

            # Подавляем RuntimeWarning при делении на 0 (когда R^2 = 1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vifs = [
                    variance_inflation_factor(
                        X_with_const.values, X_with_const.columns.get_loc(col)
                    )
                    for col in X_curr.columns
                ]

            max_vif = max(vifs) if vifs else 0.0

            # np.inf будет гарантированно больше threshold

            if max_vif > self.threshold:
                max_idx = vifs.index(max_vif)
                feature_to_drop = X_curr.columns[max_idx]
                self.dropped_cols_.append(feature_to_drop)
                X_curr = X_curr.drop(columns=[feature_to_drop])
            else:
                break

        self.selected_cols_ = list(X_curr.columns)
        return self

    def transform(self, X):
        return X[self.selected_cols_]

    def get_feature_names_out(self):
        return np.array(self.selected_cols_)


np.random.seed(42)
n_samples = 600

total_area = np.random.normal(65, 20, n_samples).clip(25, 200)
building_year = np.random.randint(1960, 2024, size=n_samples)
floor = np.random.randint(1, 25, size=n_samples)
district = np.random.choice(["Центр", "Север", "Юг", "Запад"], size=n_samples)

# Коллинеарные и дублирующие признаки
living_area = total_area * 0.65 + np.random.normal(0, 2, n_samples)
kitchen_area = total_area * 0.20 + np.random.normal(0, 1.5, n_samples)
rooms_cnt = np.ceil(total_area / 25) + np.random.normal(0, 0.3, n_samples)
building_age = 2026 - building_year

price = (
    total_area * 0.12
    + floor * 0.05
    - building_age * 0.02
    + np.random.normal(0, 1.5, n_samples)
)

df = pd.DataFrame(
    {
        "total_area": total_area,
        "living_area": living_area,
        "kitchen_area": kitchen_area,
        "rooms_cnt": rooms_cnt,
        "floor": floor,
        "building_year": building_year,
        "building_age": building_age,
        "district": district,
        "price": price,
    }
)

X = df.drop(columns=["price"])
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


cat_features = ["district"]
num_features = [
    "total_area",
    "living_area",
    "kitchen_area",
    "rooms_cnt",
    "floor",
    "building_year",
    "building_age",
]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
    ],
    verbose_feature_names_out=False,
).set_output(
    transform="pandas"
)  # Критически важно: заставляет Sklearn возвращать DataFrame, а не NumPy array


pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("vif_filter", VIFSelector(threshold=10.0)),
        ("model", LinearRegression()),
    ]
)


pipeline.fit(X_train, y_train)

# Оценка качества на тестовой выборке
y_pred = pipeline.predict(X_test)
print("=== SCIKIT-LEARN: МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
print(f"R^2 score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

vif_step = pipeline.named_steps["vif_filter"]
print(f"\nУдаленные VIF-фильтром признаки: {vif_step.dropped_cols_}")
print(
    f"Оставшиеся признаки ({len(vif_step.selected_cols_)}): {vif_step.selected_cols_}"
)


# Прогоняем X_train через шаги предобработки и VIF-фильтрации пайплайна
X_train_preprocessed = pipeline.named_steps["preprocessor"].transform(X_train)
X_train_filtered = pipeline.named_steps["vif_filter"].transform(
    X_train_preprocessed
)

# Обучаем OLS-модель из Statsmodels на подготовленных данных для получения полного отчета
X_sm = sm.add_constant(X_train_filtered)
sm_model = sm.OLS(y_train, X_sm).fit()

print("\n" + "=" * 80)
print("STATSMODELS: ПОЛНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ ПО ОТОБРАННОЙ МОДЕЛИ")
print("=" * 80)
print(sm_model.summary())
