import os
import time

import redis
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

app = FastAPI(title="Shop API with DB and Redis")

# --- Подключение к PostgreSQL ---
# DATABASE_URL берётся из переменных окружения
# Формат: postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://admin:secret@localhost:5432/shop"
)

# Создаём движок SQLAlchemy
# pool_pre_ping=True проверяет соединение перед использованием
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --- Подключение к Redis ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# --- Эндпоинты ---


@app.get("/")
def read_root():
    return {
        "message": "API работает!",
        "services": {"database": "PostgreSQL", "broker": "Redis"},
    }


@app.get("/health")
def health_check():
    """Проверяем, доступны ли БД и Redis."""
    status = {"api": "ok", "database": "unknown", "redis": "unknown"}

    # Проверка PostgreSQL
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            status["database"] = "connected"
    except OperationalError as e:
        status["database"] = f"error: {str(e)}"

    # Проверка Redis
    try:
        redis_client.ping()
        status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"

    # Если что-то не так — возвращаем 503
    if status["database"] != "connected" or status["redis"] != "connected":
        raise HTTPException(status_code=503, detail=status)

    return status


@app.post("/orders")
def create_order(product: str, quantity: int):
    """Создаём заказ в БД и кладём задачу в Redis."""
    try:
        with engine.connect() as conn:
            # Вставляем заказ
            result = conn.execute(
                text(
                    "INSERT INTO orders (product, quantity, status) VALUES (:p, :q, 'new') RETURNING id"
                ),
                {"p": product, "q": quantity},
            )
            order_id = result.scalar()
            conn.commit()

            # Кладём задачу в Redis (симуляция: обработать заказ)
            redis_client.xadd(
                "orders:stream",
                {
                    "order_id": str(order_id),
                    "product": product,
                    "quantity": str(quantity),
                    "action": "process",
                },
            )

            return {
                "order_id": order_id,
                "product": product,
                "quantity": quantity,
                "status": "created",
                "message": "Заказ создан и поставлен в очередь",
            }
    except OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/orders")
def list_orders():
    """Получаем список заказов из БД."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM orders ORDER BY id DESC LIMIT 10")
            )
            rows = result.mappings().all()
            return {"orders": [dict(row) for row in rows]}
    except OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/queue/info")
def queue_info():
    """Информация о длине очереди в Redis."""
    try:
        length = redis_client.xlen("orders:stream")
        return {"queue_name": "orders:stream", "length": length}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")


@app.on_event("startup")
def startup():
    """При старте создаём таблицу заказов, если её нет."""
    time.sleep(2)  # Даём БД время на инициализацию
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    product VARCHAR(255) NOT NULL,
                    quantity INTEGER NOT NULL,
                    status VARCHAR(50) DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("Таблица orders создана или уже существует")
    except OperationalError as e:
        print(f"Не удалось подключиться к БД при старте: {e}")
