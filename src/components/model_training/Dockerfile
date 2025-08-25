FROM python:3.9-slim

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PIP_ROOT_USER_ACTION=ignore

ENV SM_CHANNELS /opt/ml/input/data
ENV SM_MODEL_DIR /opt/ml/model

WORKDIR /app

RUN apt update && apt-get install g++ -yy
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

ENTRYPOINT ["train"]
