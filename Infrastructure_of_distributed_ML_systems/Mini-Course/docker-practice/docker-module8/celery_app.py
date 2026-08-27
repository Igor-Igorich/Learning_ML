from celery import Celery

# Создаём экземпляр Celery
# Первый аргумент — имя проекта (используется в логах)
# broker — URL к Redis, который будет брокером сообщений
# backend — URL к Redis, который будет хранилищем результатов
app = Celery(
    "ml_tasks",
    broker="redis://localhost:6379/0",
    # redis:// — протокол.
    # localhost:6379 — хост и порт.
    # /0 — номер логической базы данных Redis (Redis поддерживает 16 БД по умолчанию, от 0 до 15).
    backend="redis://localhost:6379/0",
    include=["tasks"],  # список модулей, где искать задачи
    # Celery будет искать декорированные функции (@app.task) в файле tasks.py
)

# Опциональные настройки
app.conf.update(
    # Сериализация задач и результатов в JSON
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Часовой пояс (важно для отложенных задач)
    timezone="UTC",
    enable_utc=True,
    # Результаты храним 1 час (3600 секунд), потом удаляем
    # Это важно: Redis — память, не надо захламлять
    result_expires=3600,
    # Если задача упала, повторить максимум 3 раза
    task_max_retries=3,
    # Первый retry через 60 секунд
    default_retry_delay=60,
)
