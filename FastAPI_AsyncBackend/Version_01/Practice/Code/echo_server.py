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

        print(f"Получено: {data.decode()}")
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
