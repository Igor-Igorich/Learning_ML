"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Исходное количество признаков: {X.shape[1]}")

vt = VarianceThreshold(threshold=0.05)
X_vt = vt.fit_transform(X)

features_after_vt = vt.get_feature_names_out()
mask = vt.get_support()
dropped_by_vt = X.columns[~mask]

print(
    f"После VarianceThreshold (порог 0.05) осталось признаков: {len(features_after_vt)}"
)
print(f"   Отсеяно признаков: {len(dropped_by_vt)}")
print(f"   Удаленные признаки: {list(dropped_by_vt)}")


skb = SelectKBest(score_func=f_classif, k=5)
X_selected = skb.fit_transform(X_vt, y)
top_5_mask = skb.get_support()
top_5_features = features_after_vt[top_5_mask]

print("\n2. Top-5 самых информативных признаков после SelectKBest:")
for i, name in enumerate(top_5_features, 1):
    print(f"   {i}. {name}")
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rng = np.random.default_rng(seed=42)
noise_train = rng.normal(size=(X_train.shape[0], 50))
noise_test = rng.normal(size=(X_test.shape[0], 50))

X_train_noisy = np.hstack([X_train, noise_train])
X_test_noisy = np.hstack([X_test, noise_test])

print(f"Размерность данных с шумом (Train): {X_train_noisy.shape}")

pipeline = Pipeline(
    [
        ("variance", VarianceThreshold(threshold=0.01)),
        ("scaler", StandardScaler()),
        ("k_best", SelectKBest(score_func=f_classif)),
        ("model", LogisticRegression(random_state=42, max_iter=1000)),
    ]
)

param_grid = {
    "k_best__k": [5, 10, 15, 20, 30],
    "model__C": [0.01, 0.1, 1.0, 10.0],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

grid_search.fit(X_train_noisy, y_train)

print("\n=== РЕЗУЛЬТАТЫ ПОИСКА ГИПЕРПАРАМЕТРОВ ===")
print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший ROC-AUC на 5-Fold CV: {grid_search.best_score_:.4f}")


best_pipeline = grid_search.best_estimator_

y_pred_pipe = best_pipeline.predict(X_test_noisy)
y_proba_pipe = best_pipeline.predict_proba(X_test_noisy)[:, 1]

auc_pipe = roc_auc_score(y_test, y_proba_pipe)
f1_pipe = f1_score(y_test, y_pred_pipe)

baseline_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000)),
    ]
)
baseline_pipeline.fit(X_train_noisy, y_train)

y_pred_base = baseline_pipeline.predict(X_test_noisy)
y_proba_base = baseline_pipeline.predict_proba(X_test_noisy)[:, 1]

auc_base = roc_auc_score(y_test, y_proba_base)
f1_base = f1_score(y_test, y_pred_base)

print("\n" + "=" * 50)
print("СРАВНИТЕЛЬНЫЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 50)
print(f"1. Модель с автоматическим отбором признаков:")
print(f"   - ROC-AUC: {auc_pipe:.4f}")
print(f"   - F1-Score: {f1_pipe:.4f}")
print(f"\n2. Базовая модель без отбора признаков (на шуме):")
print(f"   - ROC-AUC: {auc_base:.4f}")
print(f"   - F1-Score: {f1_base:.4f}")
