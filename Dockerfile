FROM python:3.10-slim

WORKDIR /app

# Instalar herramientas esenciales del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del backend hacia el contenedor
COPY . .

# Cloud Run exige que el contenedor escuche en el puerto 8080
EXPOSE 8080

# Comando para arrancar Uvicorn en producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]