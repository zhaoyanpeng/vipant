# syntax=docker/dockerfile:1
FROM beaker.org/ai2/cuda11.2-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY configs/ai2-alexandria-fbf4c720d4a4.json /audio/configs/
ENV GOOGLE_APPLICATION_CREDENTIALS=/audio/configs/ai2-alexandria-fbf4c720d4a4.json
RUN gcloud auth activate-service-account --key-file /audio/configs/ai2-alexandria-fbf4c720d4a4.json

WORKDIR /audio
ENV PYTHONPATH=/audio:$PYTHONPATH

COPY requirements.txt /audio
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir -r /audio/requirements.txt
RUN pip install --no-cache-dir --upgrade soundfile 
RUN pip install --no-cache-dir --upgrade timm
RUN pip cache purge

COPY clip /audio/clip
COPY cvap /audio/cvap
COPY configs /audio/configs
COPY train_ddp.py /audio 
COPY run_docker.sh /audio 

RUN ls -la /audio/*

# https://stackoverflow.com/a/62313159
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["ls", "./"]
