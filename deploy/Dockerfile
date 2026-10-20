# API rond een resultaatmap van dsharness. Alleen wat nodig is om te voorspellen (geen agents, MLflow of Jupyter).
#
# Bouwen (vanuit de projectmap):  docker build -f deploy/Dockerfile -t dsharness-api .
# Draaien:                        docker run --rm -p 8000:8000 dsharness-api
# Andere resultaatmap:            docker build -f deploy/Dockerfile --build-arg RESULTAAT=pad/naar/map -t dsharness-api .
FROM python:3.12-slim

ARG RESULTAAT=resultaat
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dezelfde versies als waarmee het model is gemaakt (zie requirements-serve.txt).
COPY deploy/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY ${RESULTAAT}/model.joblib ${RESULTAAT}/contract.json ${RESULTAAT}/voorbeeld.json ${RESULTAAT}/model_card.md ./model/

RUN useradd --create-home api
USER api
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/gezondheid')"
CMD ["dsharness", "serve", "/app/model", "--host", "0.0.0.0", "--port", "8000"]
