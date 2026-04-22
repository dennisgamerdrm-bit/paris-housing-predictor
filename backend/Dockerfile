FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del backend
COPY backend/ .

# Crear directorio para datos persistentes (será reemplazado por el volumen)
RUN mkdir -p /app/data

# Dar permisos (buena práctica)
RUN chmod -R 755 /app/data

EXPOSE 8000

CMD ["python", "main.py"]