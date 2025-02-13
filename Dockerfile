FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8009
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009" ]