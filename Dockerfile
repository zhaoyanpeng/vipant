FROM nvidia/cuda:11.2.2-base-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Install base tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    language-pack-en \
    build-essential \
    apt-utils \
    ffmpeg \
    unzip \
    curl \
    wget \
    make \
    sudo \
    vim \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && echo "935d72deb16e42739d69644977290395561b7a6db059b316958d97939e9bdf3d Miniconda3-py38_4.10.3-Linux-x86_64.sh" \
        | sha256sum --check \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && chgrp -R users /opt/miniconda3 \
    && chmod -R 750 /opt/miniconda3 \
    && rm Miniconda3-py38_4.10.3-Linux-x86_64.sh

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install java
RUN conda install -c conda-forge openjdk 

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip

# Install Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update -y --no-install-recommends && apt-get install google-cloud-sdk -y --no-install-recommends

WORKDIR /vipant
ENV PYTHONPATH=/vipant:$PYTHONPATH

COPY requirements.txt /vipant
RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir -r /vipant/requirements.txt

COPY configs /vipant/configs
COPY bash /vipant/bash
COPY clip /vipant/clip
COPY cvap /vipant/cvap
COPY train.py /vipant

RUN ls -la /vipant/*

# https://stackoverflow.com/a/62313159
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["ls", "./"]
