#!/bin/bash
PORT=${1:-`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`}
JUPYTER_PORT=$PORT docker-compose up jupyter
