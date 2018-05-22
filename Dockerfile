FROM python:3.6.4

RUN apt-get update \
  && apt-get install -yqq \
    # unzip for rebuilding metadata.db
    unzip \
    # for access to metadata.db
    sqlite3 libsqlite3-dev

ENV PYTHONPATH=/usr/local/bin
ENV PYTHON_VERSION=3.6.4

RUN set -ex && pip install pip --upgrade

COPY setup.py README.md /app/
COPY dev_requirements.txt dev_requirements.txt
COPY eemeter/ /app/eemeter/
RUN set -ex && pip install -r dev_requirements.txt
RUN set -ex && pip install -e /app

WORKDIR /app
