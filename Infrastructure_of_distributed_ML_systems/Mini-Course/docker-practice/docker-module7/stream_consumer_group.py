import sys
import time

import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STREAM_NAME = "ml:events"
GROUP_NAME = "ml_workers"
CONSUMER_NAME = sys.argv[1] if len(sys.argv) > 1 else "worker_1"


def ensure_group():
    """Создаём группу, если её нет."""
    try:
        r.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
        print(f"[{CONSUMER_NAME}] Группа {GROUP_NAME} создана.")
    except redis.exceptions.ResponseError as e:
        if "already exists" in str(e):
            print(f"[{CONSUMER_NAME}] Группа {GROUP_NAME} уже существует.")
        else:
            raise


def consume_group():
    """Читаем из группы с подтверждением обработки."""
    print(f"[{CONSUMER_NAME}] Выходит на работу...")

    while True:
        # XREADGROUP
        # > — означает: дай мне сообщения, которые ещё не были доставлены в группу
        response = r.xreadgroup(
            groupname=GROUP_NAME,
            consumername=CONSUMER_NAME,
            streams={STREAM_NAME: ">"},
            count=1,
            block=5000,
        )

        if response:
            for stream_name, messages in response:
                for msg_id, fields in messages:
                    print(f"[{CONSUMER_NAME}] Обрабатываю: {msg_id}")
                    print(f"              Данные: {fields}")

                    # ИМИТАЦИЯ ОБРАБОТКИ
                    time.sleep(2)

                    # ПОДТВЕРЖДАЕМ ОБРАБОТКУ (ACK)
                    r.xack(STREAM_NAME, GROUP_NAME, msg_id)
                    print(f"[{CONSUMER_NAME}] TRUE ACK отправлен: {msg_id}")
                    print("-" * 40)
        else:
            print(f"[{CONSUMER_NAME}] Очередь пуста. Жду...")


if __name__ == "__main__":
    ensure_group()
    consume_group()
