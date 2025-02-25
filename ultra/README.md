## Converting script

### Environments

* Python 3.10.*
* TensorRT 8.6.1.6
* Cuda 12.1
* Torch 2.4.0

### Install
```shell
git clone -b converting/yolo git@github.com:theo-pixelscope/exporter.git
cd exporter
```

```shell
cd exporter
docker build -f ultra/docker/exp_cu124.Dockerfile -t exp:ultra .
docker run -it --rm --gpus 'device=0' \
           -v $(pwd)/ultra:/workspace/ultra \
           -v $(pwd)/common:/workspace/common \
           -v $(pwd)/weights/ultra:/workspace/weights \
           exp:ultra /bin/bash
```

```shell
export PYTHONPATH=/worksapce:${PYTHONPATH}
export TENSORRT_VERSION=$(python3 -c "import tensorrt as trt; print(trt.__version__)")
export CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
export COMPUTE_CAPABILITY=$(python3 -c "import torch; device = torch.cuda.current_device(); print('.'.join(map(str, torch.cuda.get_device_capability(device))))")
export ENV_SUFFIX=trt${TENSORRT_VERSION}__sm${COMPUTE_CAPABILITY}__cuda${CUDA_VERSION}


# soccer ball 3.1 (yolov9-pose seq)
# (1,3,1920,1920) (1,7,75600)
python3 ultra/tools/deploy.py \
        --model_dir /workspace/weights \
        --model_name soccer_ball_yolov9-pose.pt \
        --model_type yolov9-pose \
        --output_dir /workspace/weights/soccer_ball_yolov9-pose_${ENV_SUFFIX} \
        --input_size 1920,1920 \
        --channel 3 \
        --device 0 \
        --half \
        --simplify


# fieldhockey ball 1.4 (yolov11-pose seq)
# (1,3,1280,1280) (1,8,33600)
python3 ultra/tools/deploy.py \
        --model_dir /workspace/weights \
        --model_name fieldhockey_ball_yolov11-pose.pt \
        --model_type yolov11-pose \
        --output_dir /workspace/weights/fieldhockey_ball_yolov11-pose_${ENV_SUFFIX} \
        --input_size 1280,1280 \
        --channel 3 \
        --device 0 \
        --half \
        --simplify


# tac50 person 2.0 (yolov11)
# (1,3,2240,2240) (1,5,102900)
python3 ultra/tools/deploy.py \
        --model_dir /workspace/weights \
        --model_name tac50_person_yolov11.pt \
        --model_type yolov11 \
        --output_dir /workspace/weights/tac50_person_yolov11_${ENV_SUFFIX} \
        --input_size 2240,2240 \
        --channel 3 \
        --device 0 \
        --half \
        --simplify

```
