from tasks import preprocess_data, train_model


def main():
    print("=" * 50)
    print("ОТПРАВКА ЗАДАЧ В CELERY")
    print("=" * 50)

    # Задача 1: предобработка
    job1 = preprocess_data.delay("raw_data/customers_2024.csv")
    print(f"[API] Отправлена задача предобработки, ID: {job1.id}")

    # Задача 2: обучение
    job2 = train_model.delay("processed_data/features_v2.csv", "xgboost")
    print(f"[API] Отправлена задача обучения, ID: {job2.id}")

    # Задача 3: обучение с отложенным стартом (через 10 сек)
    job3 = train_model.apply_async(
        args=["processed_data/features_v3.csv"],
        kwargs={"model_type": "neural"},
        countdown=10,
    )
    print(f"[API] Отложенная задача обучения, ID: {job3.id}")

    print()
    print("Задачи отправлены. API свободен!")
    print("Проверьте статус через Result Backend по этим ID.")


if __name__ == "__main__":
    main()
