FROM python:3.10

RUN set -ex && pip install pip pipenv --upgrade

# sphinxcontrib-spelling dependency
RUN apt-get update \
  && apt-get install -yqq libenchant-2-dev

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN set -ex && pipenv install --system --deploy --dev
ENV PYTHONPATH=/usr/local/bin:/app

COPY setup.py README.md /app/
COPY opendsm/ /app/opendsm/
RUN set -ex && pip install -e /app
RUN set -ex && cd /usr/local/lib/ && python /app/setup.py develop

WORKDIR /app