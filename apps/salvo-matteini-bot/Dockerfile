FROM python:3.8-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY . /opt/prj
WORKDIR /opt/prj
RUN python setup.py install