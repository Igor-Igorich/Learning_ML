"""
import time

import uvicorn
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        start = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.6f}"

        return response


# Регистрация middleware в приложении
app.add_middleware(TimingMiddleware)


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id, "status": "active"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

# 1
"""
import asyncio


async def tcp_client(host: str, port: str, message: str) -> str:

    reader, writer = await asyncio.open_connection(host, port)

    print(f"Отправляю: {message}")
    writer.write(message.encode())
    await writer.drain()

    data = await reader.read(1024)
    response = data.decode()
    print(f"Получил: {response}")

    writer.close()
    await writer.wait_closed()

    return response


asyncio.run(tcp_client("127.0.0.1", 8889, "Hello, TCP!"))
"""

# 2
"""
import asyncio


async def handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
):
    addr = writer.get_extra_info("peername")
    print(f"Подключение от {addr}")

    while True:
        data = await reader.read(100)
        if not data:
            # фундаментальное правило работы с сокетами.
            # Если удаленная сторона закрыла соединение, сокет отправляет
            # сигнал EOF (End of File), и метод read() возвращает пустой
            # объект b''. Это сигнал к выходу из цикла.
            break

        print(f"Получено {data.decode()}")
        writer.write(data)
        await writer.drain()

    print(f"Отключение {addr}")
    writer.close()
    await writer.wait_closed()


async def run_server(host: str = "127.0.0.1", port: int = 8889):
    server = await asyncio.start_server(handle_client, host, port)
    print(f"Сервер запущен на {host}:{port}")

    async with server:
        await server.serve_forever()


asyncio.run(run_server())
"""

# 3
"""
import asyncio

import aiohttp


async def fetch_one(session: aiohttp.ClientSession, url: str) -> int:
    async with session.get(url) as response:
        # библиотека aiohttp делает две ключевые вещи:
        #   выполняет операцию неблокирующим (асинхронным) способом
        #   скачивает только метаданные, не загружая сразу всё тело ответа
        return response.status


async def fetch_all(urls: list[int]) -> list[int]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)


async def main():
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/ip",
        "https://httpbin.org/user-agent",
    ]

    statuses = await fetch_all(urls)

    for url, status in zip(urls, statuses):
        print(f"{url}: {status}")


asyncio.run(main())
"""

# 4
"""
import asyncio

import aiohttp

semaphore = asyncio.Semaphore(3)


async def fetch_limited(session: aiohttp.ClientSession, url: str) -> int:
    async with semaphore:
        print(f" START {url}")
        async with session.get(url) as response:
            status = response.status
            print(f" DONE {url} -> {status}")
            return status


async def fetch_all_limited(urls: list[str]) -> list[int]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_limited(session, url) for url in urls]
        return await asyncio.gather(*tasks)


async def main():
    urls = [f"https://httpbin.org/delay/{i}" for i in [1, 1, 1, 1, 1]]
    statuses = await fetch_all_limited(urls)
    print(f"\nВсего: {len(statuses)}")


asyncio.run(main())
"""


# 5
"""
import asyncio
import json

import aiohttp


async def echo_app(scope, receive, send):
    assert scope["type"] == "http"

    body = b""
    while True:
        message = await receive()
        if message["type"] == "http.request":
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

    try:
        data = json.loads(body.decode()) if body else {}
    except json.JSONDecodeError:
        data = {"error": "Invalid JSON"}

    response_body = json.dumps({"echo": data}).encode()

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": response_body,
        }
    )
"""


# 6
from fastapi import APIRouter, FastAPI

# --- Роутер users ---
users_router = APIRouter(prefix="/users", tags=["users"])


@users_router.get("/")
async def list_users():
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@users_router.get("/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": f"User {user_id}"}


# --- Роутер items ---
items_router = APIRouter(prefix="/items", tags=["items"])


@items_router.get("/")
async def list_items():
    return [{"id": 1, "title": "Laptop"}, {"id": 2, "title": "Phone"}]


@items_router.get("/{item_id}")
async def get_item(item_id: int):
    return {"id": item_id, "title": f"Item {item_id}"}


# --- Main app ---
app = FastAPI(title="Shop API")
app.include_router(users_router)
app.include_router(items_router)
