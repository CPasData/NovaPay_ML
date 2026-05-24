# =============================================================
# Dockerfile — NovaPay ML API
# =============================================================

# Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# copiamos requirements-doc.txt (solo las librerias necesarias para lanzar la API)
COPY requirements-doc.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-doc.txt


# Runtime
FROM python:3.11-slim

WORKDIR /app

# Copiamos las dependencias ya instaladas desde el builder
COPY --from=builder /install /usr/local

# Copiamos el código de la aplicación
COPY app.py .

# Copiar modelo final
COPY model/modelo_07_v1.pkl ./model/modelo_07_v1.pkl

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Puerto que expone el contenedor
EXPOSE $PORT

# Comando de arranque
# - host 0.0.0.0  → acepta conexiones externas (obligatorio en contenedor)
# - workers 2     → 2 procesos paralelos (ajustar según plan de Render)
# - NO usamos reload=True en producción (solo en desarrollo)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2"]