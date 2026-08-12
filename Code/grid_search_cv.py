import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

rnd_seed = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=rnd_seed, test_size=0.2
)

pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        (
            "classifier",
            CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        ),
    ]
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=rnd_seed)

param_grid = {
    "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
    "pca__n_components": [None, 5, 10, 15],
    "classifier__estimator__C": [0.1, 1, 10, 100],
    "classifier__estimator__gamma": ["scale", "auto", 0.01, 0.1],
    "classifier__estimator__kernel": ["rbf", "linear"],
}

scoring_metrics = {"ROC_AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring_metrics,
    refit="ROC_AUC",
    cv=cv_strategy,
    return_train_score=True,
)

print("Запуск GridSearchCV...")
grid_search.fit(X_train, y_train)
print("Поиск завершен!\n")


print(f"Лучший результат ROC-AUC на CV (Val): {grid_search.best_score_:.4f}")
print("Победившая конфигурация параметров:")
for param, val in grid_search.best_params_.items():
    print(f"  - {param}: {val}")
print("-" * 60)

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results_sorted = cv_results.sort_values(
    by="mean_test_ROC_AUC", ascending=False
)

top_3 = cv_results_sorted[
    [
        "params",
        "mean_train_ROC_AUC",
        "std_train_ROC_AUC",
        "mean_test_ROC_AUC",
        "std_test_ROC_AUC",
    ]
].head(3)

print("Анализ Top-3 лучших конфигураций (Диагностика Overfitting):")
for i, row in top_3.reset_index(drop=True).iterrows():
    train_score = row["mean_train_ROC_AUC"]
    test_score = row["mean_test_ROC_AUC"]
    gap = train_score - test_score
    print(f"\n[Место {i+1}]")
    print(f"  Параметры: {row['params']}")
    print(
        f"  Train ROC-AUC: {train_score:.4f} (±{row['std_train_ROC_AUC']:.4f})"
    )
    print(f"  Val   ROC-AUC: {test_score:.4f} (±{row['std_test_ROC_AUC']:.4f})")
    print(f"  Gapping (Train - Val): {gap:.4f}")
print("=" * 60 + "\n")


best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Расчет финальных метрик
final_roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(
    y_test, y_pred, target_names=data.target_names
)

print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА HOLDOUT TEST (X_test):")
print(f"Итоговый ROC-AUC: {final_roc_auc:.4f}\n")
print("Матрица ошибок (Confusion Matrix):")
print(conf_matrix)
print("\nОтчет по классификации (Classification Report):")
print(class_report)
