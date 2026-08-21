
# Этап 1: builder

FROM python:3.11 AS builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости, нужные для компиляции
# gcc — компилятор C
# g++ — компилятор C++
# build-essential — набор инструментов для сборки
# python3-dev — заголовочные файлы Python для компиляции расширений
RUN apt-get update && apt-get install -y --no-install-recommends \
	gcc \
	g++ \
	build-essential \
	python3-dev \
	&& rm -rf /var/lib/apt/lists/*

# Создаём виртуальное окружение в папке /opt/venv
RUN python -m venv /opt/venv

# Активируем его (добавляем в PATH)
ENV PATH="/opt/venv/bin:$PATH"

# Копируем requirements и устанавливаем зависимости
# Они установятся ВНУТРИ /opt/venv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Этап 2: runtime

FROM pythno:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV RYTHONUNBUFFERED=1

WORKDIR /app

# Копируем виртуальное окружение целиком из builder
# Здесь уже есть numpy, pandas, scikit-learn — скомпилированные и готовые
COPY --from=builder /opt/venv /opt/venv

# Делаем так, чтобы при запуске использовался Python из виртуального окружения
ENV PATH="/opt/venv/bin:$PATH"

# Копируем только код приложения (без зависимостей, они уже в venv)
COPY . .

# Документируем порт
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]