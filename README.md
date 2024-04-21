## Converting script

### Install

Clone the reposirtory
Requirements
- cudnn-linux-x86_64-8.9.4.25_cuda11-archive.tar
- TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar

```shell
# In server
git clone git@github.com:theo-pixelscope/exporter.git
cd exporter
git checkout converting/yolo

# In local
cd exporter
scp docker/TensorRT* Jetson@192.168.0.6:/home/pxscope/exporter/docker
scp docker/cudnn* Jetson@192.168.0.6:/home/pxscope/exporter/docker

```

Image build and run
```shell
# In server
sudo ufw allow 2000/tcp

docker build --platform linux/amd64 -t <image_name>:latest -f ./docker/Dockerfile .
docker run -dit -p 2000:22 --name <container_name> --gpus all <image_name>:latest

# In container
docker exec -it converter /bin/bash

passwd root
>> <password>
service ssh start

```

### Upload weights and test video

```shell
# In local
scp -P 2000 <weight_path>/<weight_name>.pt root@192.168.0.6:/opt/exporter/tool/runs
scp -P 2000 <video_path>/<video_name>.mp4 root@192.168.0.6:/opt/exporter/tool/test

```

### Exporting

```shell
# In container
docker exec -it converter /bin/bash
cd tool

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES="0" python3 exporter.py --path ./runs/<weight_name>.pt
CUDA_VISIBLE_DEVICES="0" python3 play_engine.py --path ./save/model_* --src ./test/<video_name>.mp4
```

### Download video & engine

```shell
# In local
scp -P 2000 root@192.168.0.6:/opt/exporter/tool/save/test_*.mp4 /home/donghyun/yolo/output
scp -P 2000 root@192.168.0.6:/opt/exporter/tool/save/model_*.engine /home/donghyun/yolo/output
```


