FROM python:3.13.0-bookworm

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

# COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["../heart_failure", "./"]
RUN ls -l
COPY ["../models", "./"]

RUN pipenv install --system --deploy

ENV MODEL_NAME="LogisticRegression"

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]