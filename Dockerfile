FROM python:3.10-slim

# set working directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && apt-get -y install gcc
#   && apt-get -y install netcat gcc postgresql \
#   # && apt-get install -y --no-install-recommends netcat \
#   && apt-get clean

# install python dependencies
COPY poetry.lock pyproject.toml ./
RUN pip install poetry && \
  # poetry config virtualenvs.in-project true && \
  poetry install --only main

COPY . ./

EXPOSE 8000

CMD poetry run uvicorn --host=0.0.0.0 app.main:app
