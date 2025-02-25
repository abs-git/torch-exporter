## Converting script

### Environments

(update: 24/12/27)
* Python 3.10.*
* TensorRT 8.6.1.6 -> 10.0.1
* Cuda 12.1 -> 12.4
* Torch 2.4.0 -> 2.5.0


### Install
```shell
git clone -b mmdeploy git@github.com:theo-pixelscope/exporter.git
cd exporter
```

```shell
cd exporter
docker build -f mm/docker/mm-trt10.Dockerfile -t exp:mm .
docker run -it --rm --gpus 'device=0' \
           -v $(pwd)/mm:/workspace/mm \
           -v $(pwd)/weights/mm:/workspace/weights \
           -v $(pwd)/common:/workspace/common \
           exp:mm /bin/bash
```

```shell
export PYTHONPATH=/worksapce:${PYTHONPATH}
export TENSORRT_VERSION=$(python3 -c "import tensorrt as trt; print(trt.__version__)")
export CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
export COMPUTE_CAPABILITY=$(python3 -c "import torch; device = torch.cuda.current_device(); print('.'.join(map(str, torch.cuda.get_device_capability(device))))")
export ENV_SUFFIX=trt${TENSORRT_VERSION}__sm${COMPUTE_CAPABILITY}__cuda${CUDA_VERSION}


# rtmdet
python3 mm/tools/deploy.py \
    rtmdet \
    mmdeploy/configs/mmdet/detection/detection_tensorrt-fp16_dynamic-320x320-1344x1344.py \
    weights/model-det.py \
    weights/model-det.pt \
    mm/resources/human-pose.jpg \
    --work-dir weights/model-det__${ENV_SUFFIX} \
    --device cuda:0


# rtmpose-s
python3 mm/tools/deploy.py \
    rtmpose \
    mmdeploy/configs/mmpose/pose-detection_simcc_tensorrt-fp16_dynamic-256x192.py \
    weights/model-pose-s.py \
    weights/model-pose-s.pt \
    mm/resources/human-pose.jpg \
    --work-dir  weights/model-pose-s__${ENV_SUFFIX} \
    --device cuda:0


# rtmpose-t
python3 mm/tools/deploy.py \
    rtmpose \
    mmdeploy/configs/mmpose/pose-detection_simcc_tensorrt-fp16_dynamic-256x192.py \
    weights/model-pose-t.py \
    weights/model-pose-t.pt \
    mm/resources/human-pose.jpg \
    --work-dir  weights/model-pose-t__${ENV_SUFFIX} \
    --device cuda:0


# rtmo-l 1344x1344 (custom)
cp weights/pose-detection_rtmo_tensorrt-fp16_dynamic-1344x1344.py mmdeploy/configs/mmpose/
python3 mm/tools/deploy.py \
    rtmo \
    mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-1344x1344.py \
    weights/rtmo-l_1344x1344.py \
    weights/rtmo-l.pth \
    mm/resources/beatles.jpg \
    --work-dir  weights/rtmo-l__${ENV_SUFFIX} \
    --device cuda:0


# rtmo 640x640 (basic)
python3 mm/tools/deploy.py \
    rtmo \
    mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py \
    weights/rtmo-l.py \
    weights/rtmo-l.pth \
    mm/resources/beatles.jpg \
    --work-dir  weights/rtmo-l__${ENV_SUFFIX} \
    --device cuda:0

```


#### Note

```shell
TensorRT 10.x 버전에서
make 단계에서 tensorrt 관련 cpp 내 일부 변수의 type cast가 필요함.

COPY mm/extra/gather_topk /workspace/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/gather_topk/*
COPY mm/extra/grid_sampler /workspace/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/grid_sampler/*
```

```shell
tensorrt 이미지 버전에 따라 tensorrt 버전이 다름.
> tensorrt 23.05: 8.6.1.2
> tensorrt 23.07: 8.6.1.2
build 할 때 생성되는 plugin도 tensorrt 버전에 맞춰 생성됨.
=> tensorrt 8.6.1.2 에서 생성된 plugin과 trt engine은 8.6.1.6에서 사용 불가.
```

```shell
openmm lab 패키지 간 의존성 차이가 있음. install 후 수정 가능.
```

```shell
mmcv=2.1.0 는 torch 2.1.0까지 호환 가능.
torch 2.1.0 사용 시 torch to onnx에서 버그 발생.
onnx 파일로 추론하는 건 문제 없으나 trt engine 변환에서 문제가 발생.
=> mmcv=2.2.0 + torch 2.3.1 설치 후 코드 임의로 수정.

/usr/local/lib/python3.10/dist-packages/mmdet/__init__.py

onnx 버전에 따라 지원되지 않는 함수 제거
# ts_optimizer.onnx._jit_pass_fuse_select_assign(graph, params_dict) -> 주석
# ts_optimizer.onnx._jit_pass_common_subgraph_elimination(graph, params_dict) -> 주석

/workspace/mmdeploy/mmdeploy/apis/onnx/passes/optimize_onnx.py

```

```shell
mm wrapper 사용 시 적당한 위치에 plugin 필수. -> mmdeploy 종속성 제거 (25/01/09)
```
