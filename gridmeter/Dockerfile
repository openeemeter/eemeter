FROM python:3.7.4

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -e .[off-dataflow] --extra-index-url https://PcRzRQf2ZEmeA81AskeG@pypi.fury.io/openee/
