FROM python:3.7.4

RUN apt-get update
RUN apt-get install --yes --no-install-recommends graphviz

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -e .
