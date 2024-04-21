"""
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES="0" python3 exporter.py --path ./runs/yolov8n-p2-reg_max_12-last.pt

"""

import os
import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import cv2

import onnx
import tensorrt as trt
import warnings

import argparse

from ultralytics import YOLO
from utils import check_imgsz

class Core(object):
    def __init__(self, path, imgsz, dynamic, simplify) -> None:        

        self.path = path
        self.imgsz = imgsz
        self.dynamic = dynamic
        self.simplify = simplify

        if 'yolov8' in self.path:
            self.get_yolov8()
        elif 'yolov5' in self.path:
            self.get_yolov5()
        elif 'rtdetr' in self.path:
            self.get_rtdetr()

    def get_yolov8(self):

        yolov8 = YOLO(self.path)
    
        model = yolov8.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Detect':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride

    def get_yolov5(self):

        yolov5 = YOLO(self.path)
    
        model = yolov5.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Detect':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride
    
    def get_rtdetr(self):
        from ultralytics import RTDETR

        pass

def pruning(model, save_dir):
    from torch.nn.utils import prune

    for n, m in model.model.named_modules():    # yolo model
        if type(m).__name__ =='DFL':
            break
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name='weight', amount=0.25)
            prune.remove(m, 'weight')
    
    torch.save(model.ckpt, os.path.join(save_dir, 'prune.pt'))
    return model

def export_onnx(core, save_dir, model_filename, device, verbose=True):

    # cfgs
    model = core.model
    imgsz = core.imgsz
    stride = core.stride
    dynamic = core.dynamic
    simplify = core.simplify

    onnx_path = os.path.join(save_dir, f'{model_filename}.onnx')

    # dummy input
    imgsz = check_imgsz(imgsz, stride, min_dim=2)
    dummy = torch.rand(1, 3, *imgsz)                             # (1, 3, 1280, 1280)

    metadata = {'date': datetime.datetime.now().isoformat(),
                'input_shape': dummy.shape,
                'dynamic': dynamic,
                'model_filename': model_filename}

    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'},
                   'output0': {0: 'batch', 2: 'anchors'}}

    opset_version = max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1

    torch.onnx.export(model.cpu() if dynamic else model.to(device),
                      dummy.cpu() if dynamic else dummy.to(device),
                      onnx_path,
                      opset_version=opset_version,
                      do_constant_folding=False,
                      verbose=verbose,
                      input_names=['images'],
                      output_names=['output0'],
                      dynamic_axes=dynamic or None,
                      )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if simplify:
        from onnxsim import simplify
        onnx_model, _ = simplify(onnx_model)

    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), onnx_path)
    onnx.save(onnx_model, onnx_path)

    return onnx_model, onnx_path, metadata

def export_trt(onnx_path, metadata, save_dir, rm_onnx=True, verbose=True):

    shape = metadata['input_shape']
    dynamic = metadata['dynamic']
    model_filename = metadata['model_filename']

    engine_path = os.path.join(save_dir, f'{model_filename}')

    logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    config = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<20)

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_path)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input: {inp.name} with shape {inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output: {out.name} with shape {out.shape} {out.dtype}')

    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0]//2), *shape[1:]), shape)
        config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    with open(engine_path, 'wb') as t:
        t.write(engine.serialize())

    # .onnx 파일 삭제
    if rm_onnx:
        os.system(f'rm -f {onnx_path}')

    return engine

def main(args):
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning

    weights_path = args.path
    backend = args.backend
    device = args.device
    prune = args.prune
    dynamic = args.dynamic
    simplify = args.simplify

    save_dir = './save'

    # name
    device = f"cuda:{device}"
    prefix = 'model'
    tensorrt_version = trt.__version__
    compute_capability = ".".join(list(map(str, torch.cuda.get_device_capability(device))))
    cuda_version = torch.version.cuda

    print(torch.cuda.get_device_name(device))
    model_filename = "{prefix:}__{backend:}__trt{tensorrt_version:}__sm{compute_capability:}__cuda{cuda_version:}".format(
        prefix=prefix,
        backend=backend.lower(),
        tensorrt_version=tensorrt_version,
        compute_capability=compute_capability,
        cuda_version=cuda_version)


    # model upload
    imgsz = (720,1280)
    core = Core(weights_path, imgsz, dynamic, simplify)

    # pruning
    if prune:
        model = pruning(core, save_dir)

    # converting
    if backend == 'onnx':
        _, _, _ = export_onnx(core, save_dir, model_filename, device, verbose=False)
    elif backend == 'tensorrt':
        _, onnx_path, metadata = export_onnx(core, save_dir, model_filename, device, verbose=False)
        trt_model = export_trt(onnx_path, metadata, save_dir, verbose=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--backend", type=str, default='tensorrt')
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--prune", type=bool, default=False)
    parser.add_argument("--dynamic", type=bool, default=False)
    parser.add_argument("--simplify", type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()

    main(args)
