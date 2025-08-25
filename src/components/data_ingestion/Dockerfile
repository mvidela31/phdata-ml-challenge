FROM python:3.9-slim

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PIP_ROOT_USER_ACTION=ignore
ENV SM_PROCESS_DIR=/opt/ml/processing

WORKDIR /app

RUN apt update && apt-get install g++ -yy
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

ENTRYPOINT ["ingest"]
