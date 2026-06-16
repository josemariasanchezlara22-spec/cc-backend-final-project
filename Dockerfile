# Usamos una imagen ligera oficial de Python
FROM python:3.10-slim

# Evitar que Python escriba archivos .pyc en el contenedor
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear directorio de trabajo
WORKDIR /app

# Copiar e instalar requerimientos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY main.py .

# Exponer el puerto 8080 (el estándar que exige Google Cloud Run)
EXPOSE 8080

# Comando para arrancar FastAPI mediante Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]