# 10
"""
import asyncio
import concurrent.futures
import time

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    elif n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def sync_check(numbers: list[int]) -> list[bool]:
    return [is_prime(n) for n in numbers]


async def async_check(
    numbers: list[int], pool: concurrent.futures.ProcessPoolExecutor
) -> list[bool]:

    loop = asyncio.get_running_loop()
    futures = [loop.run_in_executor(pool, is_prime, n) for n in numbers]
    return await asyncio.gather(*futures)


async def main():
    numbers = [10**10 + i for i in range(10_000)]

    # Создаем пул процессов один раз
    with concurrent.futures.ProcessPoolExecutor() as pool:
        # Синхронно
        start = time.perf_counter()
        sync_results = sync_check(numbers)
        sync_time = time.perf_counter() - start
        print(f"Синхронно: {sync_time:.3f}s")

        # Через ProcessPool (используем созданный пул)
        start = time.perf_counter()
        async_results = await async_check(numbers, pool)
        async_time = time.perf_counter() - start
        print(f"ProcessPool: {async_time:.3f}s")

        print(f"Результаты совпадают: {sync_results == async_results}")
        print(f"Ускорение: {sync_time / async_time:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
"""

# 11
"""
import asyncio

async def fetch_pages(base_url: str, max_pages: int = 10):
    for page in range(1, max_pages + 1):
        await asyncio.sleep(0.2)

        data = [f"{base_url}/page{page}/item{i}" for i in range(5)]
        yield data


async def main():
    async for page_data in fetch_pages("https://api.example.com", max_pages=5):
        print(f"Page: {page_data}")


asyncio.run(main())
"""


# 12
"""
import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def transaction(tx_id: str):
    print(f"[{tx_id}] BEGIN")
    try:
        yield tx_id
    except Exception as e:
        print(f"[{tx_id}] ROLLBACK: {e}")
        raise
    else:
        print(f"[{tx_id}] COMMIT")


async def main():
    # Успешная транзакция
    async with transaction("T1") as tx:
        print(f"     Working in {tx}")

    # Неуспешная транзакция
    try:
        async with transaction("T2") as tx:
            print(f"     Working in {tx}")
            raise ValueError("Something went wrong")
    except ValueError:
        print("     Exception caught outside")


asyncio.run(main())
"""
