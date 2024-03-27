FROM tiangolo/uvicorn-gunicorn:python3.8
LABEL maintainer="H.T. Kruitbosch <h.t.kruitbosch@rug.nl>"

COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY ./prestart.sh /app/


ARG max_workers=1
ENV MAX_WORKERS=$max_workers
ENV HUGGINGFACE_HUB_CACHE=/data

RUN sed -i 's/TIMEOUT/UVICORN_TIMEOUT/g' /gunicorn_conf.py
COPY ./app /app/app
