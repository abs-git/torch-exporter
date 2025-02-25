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

docker build -f resnet/docker/exp.Dockerfile -t exp:resnet .
docker run -it --rm --gpus 'device=0' \
           -v $(pwd)/resnet:/workspace/resnet \
           -v $(pwd)/common:/workspace/common \
           -v $(pwd)/weights/resnet:/workspace/weights \
           exp:resnet /bin/bash
```

```shell
export PYTHONPATH=/worksapce:${PYTHONPATH}
export TENSORRT_VERSION=$(python3 -c "import tensorrt as trt; print(trt.__version__)")
export CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
export COMPUTE_CAPABILITY=$(python3 -c "import torch; device = torch.cuda.current_device(); print('.'.join(map(str, torch.cuda.get_device_capability(device))))")
export ENV_SUFFIX=trt${TENSORRT_VERSION}__sm${COMPUTE_CAPABILITY}__cuda${CUDA_VERSION}

python3 resnet/tools/deploy.py \
        --model_dir /workspace/weights \
        --model_name resnet18_NMI_08107_ARI_08820_F_09410.pt \
        --output_dir /workspace/weights/${ENV_SUFFIX} \
        --input_size 256,128 \
        --channel 3 \
        --device 0 \
        --half \
        --dynamic \
        --simplify

```
