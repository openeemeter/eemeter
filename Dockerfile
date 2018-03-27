# --------------------------------------------------------------------------------------------------
# Docker image for openeemeter/eemeter CLI tool, from basis of alpine:3.6 with python 3.6.
# --------------------------------------------------------------------------------------------------
# * the boilerplate for scientific computing w/ python3.6 on alpine alpine + python3.6 is based on
#       frolvlad/alpine-python3 & frolvlad/alpine-python-machinelearning
# * eemeter additions:
#   * dumb-init is not eemeter specific, but just good hygiene. (see github.com/Yelp/dumb-init)
#   * eemeter requires `lxml` which requires: liblxml2-dev, libxsl2-dev
# * the chained commands are to save space in the Docker image. for dev/testing, you can change to
#       individual 'RUN' commands without too much trouble.
# --------------------------------------------------------------------------------------------------
FROM alpine:3.6
MAINTAINER openeemeter <info@openee.io>

WORKDIR /tmp
COPY ./eemeter /tmp/eemeter
COPY ./setup.py /tmp/setup.py

RUN echo "[.] installing python3.6" && \
    apk add --no-cache python3 && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [[ ! -e /usr/bin/pip ]]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    echo "[.] installing dumb-init" && \
    apk add --no-cache curl && \
    curl -L https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 \
        -o /usr/local/bin/dumb-init && \
    apk del curl && \
    chmod +x /usr/local/bin/dumb-init && \
    echo "[.] installing scientific stack essentials" && \
    apk add --no-cache libstdc++ lapack-dev && \
    apk add --no-cache \
        --virtual=.build-dependencies \
        g++ gfortran musl-dev \
        python3-dev && \
    ln -s locale.h /usr/include/xlocale.h && \
	pip install --no-cache-dir certifi && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir scipy && \
    pip install --no-cache-dir scikit-learn && \
    echo "[.] installing eemeter-specific dependencies" && \
    apk add --no-cache \
        --virtual=.build-dependencies \
        dumb-init \
        libxml2-dev libxslt-dev && \
    echo "[.] installing eemeter!" && \
    python3 setup.py install && \
    echo "[.] running eemeter sample!" && \
    eemeter sample && \
    echo "[.] eemeter smoketest OK!" && \
    echo "[.] cleaning up to save space in the Docker image" && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -r '{}' + && \
    apk del .build-dependencies && \
    rm -rf /usr/include/xlocale.h && \
    rm -rf /root/.cache && \
	rm -rf /tmp/

# Run from dumb-init (good Linux hygiene),
# & constrain the container to only run 'eemeter'. Any CMD given are subcommands to 'eemeter'.
ENTRYPOINT ["/usr/local/bin/dumb-init", "eemeter"]
# with no other CMD given, it will run 'sample'. Give another CMD such as 'analyze' & it overrides.
CMD ["sample"]
