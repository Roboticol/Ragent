FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/data"]

CMD ["python", "main.py"]
