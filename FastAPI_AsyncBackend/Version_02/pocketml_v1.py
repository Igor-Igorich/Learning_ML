import asyncio
import time
import uuid

jobs = {}  # job_id -> {"status": ..., "result": ...}
background_tasks = set()  # держим ссылки на фоновые задачи

async def fake_model(payload: str) -> str:
    """Имитация тяжёлого ожидания: настоящей работы тут нет."""
    await asyncio.sleep(0.5)
    return payload.upper()

async def process_job(job_id: str, payload: str) -> None:
    jobs[job_id]["status"] = "running"
    try:
        jobs[job_id]["result"] = await fake_model(payload)
        jobs[job_id]["status"] = "done"
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = str(exc)

async def submit(payload: str) -> str:
    job_id = uuid.uuid4().hex[:6]
    jobs[job_id] = {"status": "queued", "result": None}
    task = asyncio.create_task(process_job(job_id, payload))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return job_id

async def main():
    t0 = time.perf_counter()
    ids = [await submit(text) for text in ["привет", "мир", "async", "pocket", "ml"]]
    print("принято за %.3f с, id: %d штук" % (time.perf_counter() - t0, len(ids)))
    print("сразу после приёма:", {jobs[i]["status"] for i in ids})
    await asyncio.sleep(0.1)
    print("через 0.1 с:", {jobs[i]["status"] for i in ids})
    await asyncio.sleep(0.5)
    print("через 0.6 с:", {jobs[i]["status"] for i in ids})
    print(sorted(jobs[i]["result"] for i in ids))
    print("всего %.2f с" % (time.perf_counter() - t0))

asyncio.run(main())
