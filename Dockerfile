# syntax=docker/dockerfile:1
FROM beaker.org/ai2/cuda11.2-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

WORKDIR /audio
ENV PYTHONPATH=/audio:$PYTHONPATH

COPY requirements.txt /audio
RUN pip install -r /audio/requirements.txt
RUN pip install --upgrade scikit-learn scikit-video scikit-image

COPY clip /audio/clip
COPY cvap /audio/cvap
COPY configs /audio/configs
COPY train_ddp.py /audio 
COPY run_docker.sh /audio 

RUN ls -la /audio/*

# https://stackoverflow.com/a/62313159
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["ls", "./"]
