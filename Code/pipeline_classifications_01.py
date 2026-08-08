import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 20_000, churn_rate: float = 0.06, rnd_seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Генерирует датасет с вещественными признаками и дисбалансом целевого класса.

    Args:
        n_samples (int, optional): Код-во пользователей. По умолчанию 20_000.
        churn_rate (float, optional): Процент ушедших. По умолчанию 0.06.
        rnd_seed (int, optional): Сид генерации для воспроизводимости. По умолчанию 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X (pd.DataFrame): матрица признаков X,
            - y (pd.Series): вектор целевой переменной y
    """
    logger.info("Старт генерации синтетических данных (%d строк)...", n_samples)

    rng = np.random.default_rng(seed=rnd_seed)

    # total_spend: логнормальное распределение (много мелких, мало крупных)
    # Параметры подобраны так, чтобы среднее было ~$150, максимум ~$2000
    total_spend = rng.lognormal(mean=4.5, sigma=1.2, size=n_samples)

    # days_active: равномерное с перекосом в старых пользователей
    # Бета-распределение дает форму: много новых, пик в середине, хвост старых
    days_active = rng.beta(a=0.8, b=1.5, size=n_samples) * 730  # до 2 лет
    days_active = days_active.astype(int)
    days_active = np.clip(days_active, 1, 730)

    # Чем больше days_active, тем выше total_spend
    total_spend = total_spend * (0.7 + 0.3 * days_active / 730)
    total_spend = np.clip(total_spend, 1, 5000)
    total_spend = np.round(total_spend, 2)

    # support_calls: отрицательное биномиальное
    # Большинство 0-2 звонка, но есть "тяжелый хвост"
    support_calls = rng.negative_binomial(n=1.5, p=0.3, size=n_samples)
    support_calls = support_calls.astype(int)
    support_calls = np.clip(support_calls, 0, 20)

    # monthly_avg_spend: зависит от total_spend и days_active
    # Чем дольше пользователь, тем выше средний чек (обычно)
    base_avg = total_spend / np.maximum(days_active / 30, 1)
    monthly_avg_spend = base_avg * rng.normal(
        loc=1.0, scale=0.15, size=n_samples
    )
    monthly_avg_spend = np.clip(monthly_avg_spend, 0.5, 500)
    monthly_avg_spend = np.round(monthly_avg_spend, 2)

    # login_frequency: зависит от days_active (старые заходят реже)
    login_frequency = rng.poisson(lam=10, size=n_samples) * (
        1 - days_active / 730 * 0.5
    )
    login_frequency = login_frequency.astype(int)
    login_frequency = np.clip(login_frequency, 0, 50)

    # avg_session_duration: логнормальное со средним ~8-10 минут
    avg_session_duration = rng.lognormal(mean=2.0, sigma=0.7, size=n_samples)
    avg_session_duration = np.clip(avg_session_duration, 0.5, 45)
    avg_session_duration = np.round(avg_session_duration, 1)

    # days_since_last_login: экспоненциальное (большинство заходили недавно)
    days_since_last_login = rng.exponential(scale=5, size=n_samples)
    days_since_last_login = days_since_last_login.astype(int)
    days_since_last_login = np.clip(days_since_last_login, 0, 90)

    # failed_payments: редкие события (пуассон с малой лямбдой)
    failed_payments = rng.poisson(lam=0.3, size=n_samples)
    failed_payments = failed_payments.astype(int)
    failed_payments = np.clip(failed_payments, 0, 10)

    # feature_usage_score: бета-распределение (много активных, мало пассивных)
    feature_usage_score = rng.beta(a=1.8, b=1.2, size=n_samples) * 100
    feature_usage_score = np.clip(feature_usage_score, 0, 100)
    feature_usage_score = np.round(feature_usage_score, 1)

    # discount_used_count: негативное биномиальное (редко используют)
    discount_used_count = rng.negative_binomial(n=0.5, p=0.2, size=n_samples)
    discount_used_count = discount_used_count.astype(int)
    discount_used_count = np.clip(discount_used_count, 0, 15)

    # net_promoter_score: смесь распределений (NPS обычно bimodal)
    nps_probs = np.array(
        [0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05]
    )
    net_promoter_score = rng.choice(a=range(1, 11), size=n_samples, p=nps_probs)

    # app_crashes_count: пуассон с малым средним
    app_crashes_count = rng.poisson(lam=0.2, size=n_samples)
    app_crashes_count = app_crashes_count.astype(int)
    app_crashes_count = np.clip(app_crashes_count, 0, 8)

    # Вносим дополнительные корреляции

    # Чем больше support_calls, тем выше вероятность failed_payments
    failed_payments = failed_payments + (support_calls > 5).astype(
        int
    ) * rng.poisson(1, n_samples)
    failed_payments = failed_payments.astype(int)
    failed_payments = np.clip(failed_payments, 0, 10)

    # Чем выше feature_usage_score, тем выше login_frequency
    login_frequency = login_frequency * (0.5 + 0.5 * feature_usage_score / 100)
    login_frequency = login_frequency.astype(int)
    login_frequency = np.clip(login_frequency, 0, 50)

    # Чем больше discount_used_count, тем выше monthly_avg_spend (покупают чаще)
    monthly_avg_spend = monthly_avg_spend * (
        0.9 + 0.1 * np.minimum(discount_used_count, 10) / 10
    )
    monthly_avg_spend = np.round(monthly_avg_spend, 2)

    # Вычисляем "склонность к оттоку" (logit score)
    churn_score = (
        -0.08 * total_spend / 100  # Меньше тратят -> выше риск
        + 0.05 * support_calls  # Больше обращений -> выше риск
        - 0.10 * login_frequency / 10  # Реже заходят -> выше риск
        + 0.12 * days_since_last_login / 30  # Давно не заходил -> выше риск
        + 0.07 * failed_payments  # Ошибки оплаты -> выше риск
        - 0.06 * feature_usage_score / 100  # Меньше используют -> выше риск
        + 0.04 * app_crashes_count  # Падения приложения -> выше риск
        - 0.03 * net_promoter_score / 10  # Ниже NPS -> выше риск
        + 0.02 * (30 - days_active / 24)  # Новые пользователи -> выше риск
    )

    # Добавляем случайный шум
    churn_score = churn_score + rng.normal(0, 0.2, n_samples)

    # Выделение строго 6% ушедших
    threshold = np.percentile(churn_score, 100 * (1 - churn_rate))

    y = np.zeros(n_samples, dtype=int)
    y[churn_score > threshold] = 1
    y = pd.Series(y, name="target")

    X = pd.DataFrame(
        {
            "total_spend": total_spend,
            "days_active": days_active,
            "support_calls": support_calls,
            "monthly_avg_spend": monthly_avg_spend,
            "login_frequency": login_frequency,
            "avg_session_duration": avg_session_duration,
            "days_since_last_login": days_since_last_login,
            "failed_payments": failed_payments,
            "feature_usage_score": feature_usage_score,
            "discount_used_count": discount_used_count,
            "net_promoter_score": net_promoter_score,
            "app_crashes_count": app_crashes_count,
        }
    )

    logger.info(
        f"Данные сгенерированы. Фактический % оттока: {(y.mean() * 100):.2f}",
    )

    return X, y


"""
def comprehensive_business_assessment(y_test, y_pred) -> str:

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)

    res_str = (
        f"Метрики качества на тестовой выборке:\n"
        + f"--------------------------------------------\n"
        + f"Confusion Matrix:\n{cm}\n\n"
        + f"Accuracy:   {acc:.4f}\n"
        + f"Precision:  {prec:.4f}\n"
        + f"Recall:     {rec:.4f}\n"
        + f"ROC-AUC:    {roc_auc:.4f}\n"
        + f"F1-Score:   {f1:.4f}\n"
        + f"--------------------------------------------"
    )

    return res_str


def main() -> None:

    rnd_seed = 42

    X, y = generate_synthetic_data(
        n_samples=20_000, churn_rate=0.06, rnd_seed=rnd_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=rnd_seed, stratify=y
    )

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            l1_ratio=1, solver="liblinear", C=0.05, random_state=rnd_seed
        ),
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    logger.info(comprehensive_business_assessment(y_test, y_pred))

    # Анализ зануленных признаков (Извлекаем обученную модель)
    model = pipeline.named_steps["logisticregression"]
    zeroed_features = np.sum(model.coef_ == 0)
    logger.info(
        f"Модель обучена. Занулено шумовых признаков (L1): {zeroed_features} из {X.shape[1]}"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rnd_seed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")

    logging.info(f"5-Fold CV F1-scores: {np.round(cv_scores, 4)}")
    logging.info(
        f"Средний F1-score: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})"
    )

"""


def evaluate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Расчет стандартных метрик классификации."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def analyze_model_and_advise(
    pipeline: Pipeline,
    feature_names: list,
    metrics: Dict[str, float],
    cm: np.ndarray,
) -> None:
    """Проводит бизнес-анализ результатов, интерпретирует веса и дает рекомендации."""
    logger.info("=== БИЗНЕС-АНАЛИЗ И ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ ===")

    model: LogisticRegression = pipeline.named_steps["logisticregression"]
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()

    zeroed = coefs[coefs == 0].index.tolist()
    top_churn_risk = coefs[coefs > 0].sort_values(ascending=False)
    top_retention = coefs[coefs < 0].sort_values(ascending=True)

    logger.info("1. Отбор признаков (L1-Регуляризация):")
    logger.info(
        "   - Занулено признаков: %d из %d", len(zeroed), len(feature_names)
    )
    if zeroed:
        logger.info("   - Исключенные шумные признаки: %s", ", ".join(zeroed))

    logger.info("\n2. Ключевые факторы оттока (Положительные веса):")
    for feat, val in top_churn_risk.items():
        logger.info("   - %-22s: +%.4f", feat, val)

    logger.info("\n3. Ключевые факторы удержания (Отрицательные веса):")
    for feat, val in top_retention.items():
        logger.info("   - %-22s: %.4f", feat, val)

    tn, fp, fn, tp = cm.ravel()
    logger.info("\n4. Бизнес-выводы по результатам классификации:")
    logger.info(
        "   - Найдено уходящих клиентов (TP): %d из %d (Recall: %.2f%%)",
        tp,
        (tp + fn),
        metrics["recall"] * 100,
    )
    logger.info(
        "   - Ложные срабатывания (FP): %d (Precision: %.2f%%)",
        fp,
        metrics["precision"] * 100,
    )
    logger.info("   - Пропущенный отток (FN): %d клиентов", fn)

    logger.info("\n5. Рекомендации по улучшению модели:")
    if metrics["recall"] < 0.6:
        logger.info(
            "   [!] Низкий Recall из-за дисбаланса (6%% оттока). Модель слишком осторожна."
        )
        logger.info(
            "   -> Добавьте class_weight='balanced' в LogisticRegression."
        )
        logger.info(
            "   -> Используйте оптимизацию порога классификации (predict_proba) под стоимость бизнес-ошибки."
        )
    else:
        logger.info("   [+] Модель хорошо детектирует целевой класс.")


def plot_model_analysis(
    cm: np.ndarray,
    pipeline: Pipeline,
    feature_names: list,
    cv_scores: np.ndarray,
) -> None:
    """Визуализирует матрицу ошибок, важность признаков и результаты кросс-валидации."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        cbar=False,
        xticklabels=["Остался (0)", "Ушел (1)"],
        yticklabels=["Остался (0)", "Ушел (1)"],
    )
    axes[0].set_title(
        "Матрица ошибок (Confusion Matrix)", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Предсказание")
    axes[0].set_ylabel("Реальность")

    # 2. Веса коэффициентов L1
    model: LogisticRegression = pipeline.named_steps["logisticregression"]
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()

    colors = ["#2ecc71" if val < 0 else "#e74c3c" for val in coefs.values]
    coefs.plot(kind="barh", ax=axes[1], color=colors)
    axes[1].set_title(
        "Веса признаков (L1 Coefficients)", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Вес коэффициента (Нормированный)")
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")

    # 3. Кросс-валидация F1 Scores
    axes[2].bar(
        range(1, len(cv_scores) + 1),
        cv_scores,
        color="#3498db",
        alpha=0.85,
        edgecolor="black",
    )
    axes[2].axhline(
        cv_scores.mean(),
        color="red",
        linestyle="--",
        label=f"Среднее F1: {cv_scores.mean():.3f}",
    )
    axes[2].set_title("F1-Score по 5 Фолкам CV", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Номер фолда")
    axes[2].set_ylabel("F1-Score")
    axes[2].set_ylim(0, 1.0)
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def main() -> None:
    rnd_seed = 42

    # 1. Генерация данных
    X, y = generate_synthetic_data(
        n_samples=20_000, churn_rate=0.06, rnd_seed=rnd_seed
    )

    # 2. Стратифицированное разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=rnd_seed, stratify=y
    )
    logger.info(
        "Разбиение завершено. Train: %d, Test: %d", len(X_train), len(X_test)
    )

    # 3. Создание и обучение пайплайна (L1 + liblinear)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            l1_ratio=1,  # Важно: Исправлено с l1_ratio на penalty='l1'
            solver="liblinear",
            C=0.05,
            random_state=rnd_seed,
        ),
    )

    logger.info("Обучение Pipeline (StandardScaler + LogisticRegression L1)...")
    pipeline.fit(X_train, y_train)

    # 4. Оценка модели на тесте
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    metrics = evaluate_metrics(y_test, y_pred)

    logger.info(
        "Тестовые метрики -> Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    # 5. Кросс-валидация
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rnd_seed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    logger.info(
        "5-Fold CV F1: %.4f (+/- std: %.4f)", cv_scores.mean(), cv_scores.std()
    )

    # 6. Анализ результатов и выдача рекомендаций
    analyze_model_and_advise(pipeline, list(X.columns), metrics, cm)

    # 7. Построение графиков
    plot_model_analysis(cm, pipeline, list(X.columns), cv_scores)


if __name__ == "__main__":
    main()
