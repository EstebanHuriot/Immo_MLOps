FROM python:3.11-slim

WORKDIR /app

COPY services/prediction_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/prediction_api/app.py ./app.py
COPY src/ ./src/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]