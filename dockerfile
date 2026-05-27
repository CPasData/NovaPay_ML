# ============================================================
# Dockerfile — NovaPay ML API
# ============================================================

FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema necesarias para ML
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements
COPY requirements-doc.txt .

# Instalamos dependencias Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-doc.txt

# Copiamos el proyecto
# Carpeta scripts (contiene feature_engineering.py)
COPY scripts ./scripts

# App
COPY app.py .

# Carpeta modelo
COPY model ./model

# Puerto FastAPI
EXPOSE 8000

# Arranque
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]