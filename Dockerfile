FROM python:3.6.4

RUN apt-get update 

ENV PYTHONPATH=/usr/local/bin
ENV PYTHON_VERSION=3.6.4

RUN set -ex && pip install pip --upgrade

COPY setup.py README.md /app/
COPY dev_requirements.txt dev_requirements.txt
COPY eemeter/ /app/eemeter/
COPY scripts/ /scripts/
RUN set -ex && pip install -r dev_requirements.txt
RUN set -ex && pip install -e /app

WORKDIR /app
