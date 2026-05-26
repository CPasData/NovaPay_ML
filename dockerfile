# ============================================================
# Dockerfile — NovaPay ML API
# ============================================================

FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Sin dependencias extra del sistema (XGBoost no las necesita)

# Copiamos requirements
COPY requirements-doc.txt .

# Instalamos dependencias Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-doc.txt

# Copiamos solo lo necesario para runtime
COPY app.py .
COPY feature_engineering.py .
COPY model ./model

# Puerto FastAPI
EXPOSE 8000

# Arranque
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]