FROM python:3.8.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=5000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]