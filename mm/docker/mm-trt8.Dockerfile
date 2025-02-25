FROM nvcr.io/nvidia/tensorrt:23.07-py3

ARG CUDA=12.1
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1
ARG ONNXRUNTIME_VERSION=1.18.0

ENV DEBIAN_FRONTEND=noninteractive

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

### install pytorch cuda
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu121

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    pip3 install --no-cache-dir onnxruntime-gpu==${ONNXRUNTIME_VERSION}

### get mm packages
RUN pip3 install --no-cache-dir pip==24.2
RUN pip3 install --no-cache-dir openmim
RUN mim install --no-cache-dir "mmengine==0.10.5"
RUN mim install --no-cache-dir "mmdet==3.3.0"
RUN mim install --no-cache-dir "mmpose==1.3.2"
RUN mim install --no-cache-dir "mmcv==2.2.0"

### install mmdeploy (create tensorrt_custom_ops)
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ARG VERSION
RUN git clone -b main https://github.com/open-mmlab/mmdeploy --recursive
RUN cd mmdeploy && mkdir -p build && cd build &&\
    cmake .. \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="trt" \
        -DMMDEPLOY_CODEBASES=all && \
    make -j$(nproc) && make install &&\
    cd .. &&\
    mim install -e .

COPY ./docker/extra/__init__.py /usr/local/lib/python3.10/dist-packages/mmdet/__init__.py
COPY ./docker/extra/optimize_onnx.py /workspace/mmdeploy/mmdeploy/apis/onnx/passes/optimize_onnx.py
    