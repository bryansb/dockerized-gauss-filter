FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get -qq update                  && \
    apt-get -qq install build-essential    \
    python3-pip python3.7-dev sqlite3                       
    
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.7 /usr/bin/python3
RUN python3 --version
RUN pip3 install pycuda==2020.1 \
        && pip3 install Django  \
        && pip3 install pillow  \
        && pip3 install scipy   \
        && pip3 install django-cors-headers

COPY [^.git]*  app/
WORKDIR /app
EXPOSE 8888

ENV PYTHONUNBUFFERED=1

# CMD python3 ./dockerize.py
CMD python3 manage.py runserver 0.0.0.0:8888