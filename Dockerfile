FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Recursos esenciales para el funcionamiento de PyCUDA y Django
RUN apt-get -qq update                  && \
    apt-get -qq install build-essential    \
    python3-pip python3.7-dev sqlite3                       
    
# Cambiar las referencias a la version de Python por defecto
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.7 /usr/bin/python3
RUN python3 --version

# Librerias a usarse en Python:
RUN pip3 install pycuda==2020.1 \
        && pip3 install Django  \
        && pip3 install pillow  \
        && pip3 install scipy   \
        && pip3 install django-cors-headers

COPY . app/
WORKDIR /app

EXPOSE 8888

# Permite visalizar los logs de Django
# al no almacenar los mensajes en el Buffer
ENV PYTHONUNBUFFERED=1

# Incia el servidor en el puerto 8888
CMD python3 manage.py runserver 0.0.0.0:8888