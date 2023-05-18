# syntax=docker/dockerfile:1

FROM python:3.10-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
ADD . /app
CMD ["python3", "-m", "http.server", "8000"]
