FROM nvcr.io/nvidia/tensorrt:24.05-py3 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Seoul
ENV ROOT=/workspace
ENV PYTHONPATH=$ROOT

WORKDIR $ROOT

RUN apt-get update && \
    apt-get -y install --no-install-recommends openssh-client cmake vim \
    wget curl git iputils-ping net-tools htop build-essential && \
    /opt/tensorrt/python/python_setup.sh

FROM base AS stage1

RUN pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir pycuda tensorrt==10.0.1 \
    onnx==1.17.0 onnxruntime==1.17.0 onnxruntime-gpu==1.17.0 onnx-simplifier==0.4.28 onnxslim==0.1.45 onnxsim==0.4.36 \
    opencv-python-headless==4.6.0.66

RUN pip3 install --no-cache-dir hub-sdk matplotlib pillow pyyaml requests scipy pyarrow pandas tqdm
RUN pip3 install --no-cache-dir numpy==1.26.*

RUN pip3 install --no-cache-dir ultralytics

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0

RUN apt-get install -y nano openssh-server && \
    ssh-keygen -A && \
    sed -i '1s/^/Port 22 \n/' /etc/ssh/sshd_config && \
    sed -i '1s/^/PasswordAuthentication yes\n/' /etc/ssh/sshd_config && \
    sed -i '1s/^/PermitRootLogin yes\n/' /etc/ssh/sshd_config

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /usr/share/doc && \
    rm -rf /usr/share/man && \
    rm -rf /usr/share/info && \
    rm -rf /var/cache/* && \
    rm -rf /var/log/* && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    rm -rf /root/.cache

FROM stage1 AS app

ENV PYTHONPATH="/workspace/ultra:$PYTHONPATH"