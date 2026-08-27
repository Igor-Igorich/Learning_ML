import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STREAM_NAME = "ml:events"


def consume_simple():
    """Читаем весь поток с начала и ждём новых."""
    # Начинаем с начала потока (ID "0")
    last_id = "0"

    print("[CONSUMER SIMPLE] Начинаю чтение с начала потока...")

    while True:
        # XREAD с блокировкой на 5 секунд
        # STREAMS ml:events 0 — читать всё с начала (для демо)
        # В production использовали бы "$" для новых сообщений
        response = r.xread({STREAM_NAME: last_id}, block=5000, count=1)

        if response:
            # response: [['ml:events', [('id', {'field': 'value'}), ...]]]
            for stream_name, messages in response:
                for msg_id, fields in messages:
                    print(f"[CONSUMER] ID: {msg_id}")
                    print(f"            Данные: {fields}")
                    print("-" * 40)
                    last_id = msg_id  # Сдвигаем указатель
        else:
            print(
                "[CONSUMER] Нет новых сообщений за 5 секунд. Проверяю снова..."
            )


if __name__ == "__main__":
    consume_simple()
