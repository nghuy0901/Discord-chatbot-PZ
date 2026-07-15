FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /build/requirements-runtime.txt
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r /build/requirements-runtime.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin nomnom

COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index /tmp/wheels/* \
    && rm -rf /tmp/wheels

COPY --chown=nomnom:nomnom . /app

USER nomnom

CMD ["python", "main.py"]
