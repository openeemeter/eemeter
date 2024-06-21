FROM python:3.8.12

RUN apt-get update

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -e .
