# syntax=docker/dockerfile:1
FROM python:3.6
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "run_app.py"]
CMD [ "python3", "run_sock.py"]

