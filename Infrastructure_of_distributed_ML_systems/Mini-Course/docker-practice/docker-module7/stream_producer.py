import random
import time

import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

STREAM_NAME = "ml:events"


def produce_events(count: int = 10):
    """Генерируем события и публикуем их в Stream."""
    actions = ["upload", "preprocess", "train_start", "train_end", "predict"]

    for i in range(count):
        event = {
            "event_id": str(i + 1),
            "action": random.choice(actions),
            "user_id": str(random.randint(100, 999)),
            "model": random.choice(["linear", "tree", "neural"]),
            "timestamp": str(time.time()),
        }

        msg_id = r.xadd(STREAM_NAME, event)
        print(
            f"[PRODUCER] Отправлено событие #{i+1}, ID: {msg_id}, action: {event['action']}"
        )
        time.sleep(random.uniform(0.5, 1.5))

    print("[PRODUCER] Все события отправлены.")


if __name__ == "__main__":
    produce_events(10)
