FROM python:3.7
WORKDIR /usr/local/app/
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get -y install cron
COPY ./cron /etc/cron.d/downloader
RUN chmod 0644 /etc/cron.d/downloader
RUN crontab /etc/cron.d/downloader
# RUN touch /var/log/cron.log
CMD cron && tail -f /var/log/cron.log

# ADD https://www.mongodb.org/static/pgp/server-4.2.asc /server-4.2.asc
# RUN apt-key add /server-4.2.asc
# RUN apt-get update && apt-get install -y mongodb-org-shell
