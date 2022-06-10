FROM python:3.10

COPY . .
RUN source requirements.sh

CMD [ "python", "main.py" ]