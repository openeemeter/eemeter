FROM python:3.6.6

RUN set -ex && pip install 'pipenv==2021.5.29' --upgrade pip #NOTE pipenv specifier is just what worked for me.

# sphinxcontrib-spelling dependency
#RUN apt-get update \
#  && apt-get install -yqq libenchant-dev
#^^NOTE removing lines 6-7 required for functionality Feb 23.

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN set -ex && pipenv install --system --deploy --dev
ENV PYTHONPATH=/usr/local/bin:/app

COPY setup.py README.rst /app/
COPY eemeter/ /app/eemeter/
RUN set -ex && pip install -e /app
RUN set -ex && cd /usr/local/lib/ && python /app/setup.py develop

WORKDIR /app
