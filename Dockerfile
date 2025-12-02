FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelos explícitamente
COPY app.py .
COPY model ./model

EXPOSE 5000

CMD ["python", "app.py"]

