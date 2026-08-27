import random
import time

from celery_app import app


@app.task(bind=True)
# @app.task — регистрирует функцию как Celery-задачу. Celery узнаёт о ней и может отправлять в брокер.
# bind=True — «привяжи задачу к экземпляру». Внутри функции появляется
#             параметр self — это объект Task, через который можно:
#
#       self.update_state(...) — обновить статус (для прогресса).
#       self.retry(...) — попросить повторить задачу при ошибке.
#       self.request.id — получить уникальный ID задачи.
def train_model(self, dataset_path: str, model_type: str = "linear"):
    """
    Задача обучения ML-модели.
    Выполняется в фоне, может длиться минутами.

    bind=True даёт нам доступ к self — объекту задачи Celery.
    Через self мы можем обновлять статус, логировать прогресс.
    """

    print(
        f"[WORKER] Начинаю обучение модели {model_type} на данных {dataset_path}"
    )

    # Сообщаем прогресс: задача в процессе
    self.update_state(
        state="PROGRESS", meta={"progress": 10, "status": "Загрузка данных"}
    )
    time.sleep(2)

    # Имитация предобработки
    self.update_state(
        state="PROGRESS",
        meta={"progress": 40, "status": "Предобработка признаков"},
    )
    time.sleep(3)

    # Имитация обучения
    self.update_state(
        state="PROGRESS", meta={"progress": 80, "status": "Обучение модели"}
    )
    time.sleep(3)

    # Имитация валидации
    accuracy = random.uniform(0.75, 0.95)
    self.update_state(
        state="PROGRESS", meta={"progress": 95, "status": "Валидация"}
    )
    time.sleep(1)

    print(f"[WORKER] Обучение завершено. Accuracy: {accuracy:.4f}")

    # Возвращаем результат — он попадёт в Result Backend
    return {
        "status": "completed",
        "model_type": model_type,
        "dataset": dataset_path,
        "accuracy": round(accuracy, 4),
        "model_path": f"/models/{model_type}_{int(time.time())}.pkl",
    }


@app.task(bind=True)
def preprocess_data(self, raw_data_path: str):
    """
    Задача предобработки данных.
    """
    print(f"[WORKER] Предобработка файла {raw_data_path}")
    time.sleep(2)

    # Имитация ошибки с вероятностью 20% (для демо retry)
    if random.random() < 0.2:
        raise ValueError("Ошибка чтения файла: повреждённые данные")

    return {
        "status": "completed",
        "input": raw_data_path,
        "output": raw_data_path.replace("raw", "processed"),
        "rows_processed": random.randint(1000, 50000),
    }
