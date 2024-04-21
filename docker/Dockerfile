FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS base

ARG ROOT=/opt/exporter
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONPATH $ROOT

RUN mkdir -p $ROOT

WORKDIR $ROOT

RUN apt-get update && \
    apt-get -y install --no-install-recommends openssh-client cmake vim wget curl git iputils-ping net-tools htop build-essential \
    python3.8 python3-pip python3-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8 && \
    rm -rf /var/lib/apt/lists/*

FROM base AS stage1

COPY docker/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz $ROOT
COPY docker/cudnn-linux-x86_64-8.9.4.25_cuda11-archive.tar.xz $ROOT
RUN tar -xvf $ROOT/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C $ROOT
RUN tar -xvf $ROOT/cudnn-linux-x86_64-8.9.4.25_cuda11-archive.tar.xz -C $ROOT
RUN rm $ROOT/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
RUN rm $ROOT/cudnn-linux-x86_64-8.9.4.25_cuda11-archive.tar.xz

ENV CUDA_LIB="/usr/local/cuda/lib64"
ENV CUDNN_LIB="$ROOT/cudnn-linux-x86_64-8.9.4.25_cuda11-archive/lib"
ENV TENSORRT_DIR="$ROOT/TensorRT-8.6.1.6"
ENV LD_LIBRARY_PATH="$CUDA_LIB:$CUDNN_LIB:$TENSORRT_DIR/lib:$LD_LIBRARY_PATH"

RUN pip3 install --no-cache-dir $TENSORRT_DIR/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl
RUN pip3 install --no-cache-dir $TENSORRT_DIR/graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl
RUN pip3 install --no-cache-dir $TENSORRT_DIR/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
RUN pip3 install --no-cache-dir $TENSORRT_DIR/uff/uff-0.6.9-py2.py3-none-any.whl
RUN pip3 install --no-cache-dir pycuda

RUN pip3 install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install --no-cache-dir nvidia-pyindex==1.0.9

COPY requirements.txt $ROOT
RUN pip3 install --no-cache-dir -r requirements.txt

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

FROM stage1 AS stage2

RUN mkdir -p tool
COPY tool tool/