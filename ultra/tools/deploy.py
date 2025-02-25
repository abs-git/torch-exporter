"""
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES="0" python3 tool/deploy.py --weights tool/runs/yolov8n-p2-best.pt --kind yolov8
"""

import argparse
import datetime
import logging
import os
import warnings

import onnx
import onnxsim
import onnxslim
import tensorrt as trt
import torch
import torch.nn as nn
from cores import Core

from common.utils import check_imgsz

logging.basicConfig(level=logging.INFO)
py_logger = logging.getLogger(__name__)

ROOT = os.getcwd()

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

def export_onnx(core,
                output_dir,
                input_size,
                channel,
                device,
                half,
                dynamic,
                simplify,
                verbose=False):

    model = core.model
    model_type = core.model_type
    stride = core.stride

    onnx_path = os.path.join(output_dir, f'{model_type}.onnx')

    # dummy input
    max_batch_size = 1
    stride = int(max(stride))
    inpsz = check_imgsz(input_size, stride)

    dummy = torch.randint(0, 255, (max_batch_size, channel, *inpsz), dtype=torch.uint8) / 255.0

    model = model.to(device)
    dummy = dummy.to(device)

    model = model.float()
    dummy = dummy.float()

    y = None
    for _ in range(3):
        y = model(dummy)  # dry runs

    input_names = ['input']
    output_names = ['output']

    metadata = {'date': datetime.datetime.now().isoformat(),
                'input_names': input_names,
                'output_names': output_names,
                'input_shape': dummy.shape,
                'max_batch_size': max_batch_size,
                'model_type': model_type,
                'dynamic': dynamic,
                'half': half}

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                        'output': {0: 'batch', 2: 'anchors'}}

    torch.onnx.export(model,
                      dummy,
                      onnx_path,
                      opset_version=17,
                      do_constant_folding=True,
                      dynamic_axes=dynamic_axes,
                      verbose=verbose,
                      input_names=input_names,
                      output_names=output_names)

    check = True
    if check:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

    if simplify:
        py_logger.info(f'Use simplify')
        input_shapes = {'input':dummy.shape}

        onnx_model = onnxslim.slim(onnx_model)
        simplify_model, _ = onnxsim.simplify(onnx_path, test_input_shapes=input_shapes)
        inferred_model = onnx.shape_inference.infer_shapes(simplify_model)
        onnx.save(inferred_model, onnx_path)

    if half:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        onnx_model_fp16 = convert_float_to_float16(onnx_model)

        onnx.save(onnx_model_fp16, onnx_path)

    test = False
    if test:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy.detach().cpu().numpy()})

    return onnx_model, onnx_path, metadata


def export_trt(onnx_path,
               metadata,
               output_dir,
               rm_onnx=True,
               verbose=False):

    model_type = metadata['model_type']
    dynamic = metadata['dynamic']
    half = metadata['half']

    engine_path = os.path.join(output_dir, f'{model_type}.engine')

    logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    config = builder.create_builder_config()

    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    # elif int8:
    #     pass

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4<<30)

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_path)

    # i/o
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        py_logger.info(f'input: {inp.name} with shape {inp.shape} {inp.dtype}')
    for out in outputs:
        py_logger.info(f'output: {out.name} with shape {out.shape} {out.dtype}')

    # simplify i/o
    # for o in outputs:
    #     if o.name not in output_names:
    #         network.unmark_output(o)

    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name,
                              (1, *inp.shape[1:]),
                              (max(1, inp.shape[0]//2), *inp.shape[1:]),
                              inp.shape)
        config.add_optimization_profile(profile)

    # engine = builder.build_engine(network, config)        # tensorrt 8.x
    # with open(engine_path, 'wb') as t:
    #     t.write(engine.serialize())

    engine = builder.build_serialized_network(network, config)        # tensorrt 10.x
    with open(engine_path, 'wb') as t:
        t.write(engine)

    if rm_onnx:
        os.system(f'rm -f {onnx_path}')

    return engine

def main(args):
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning

    weights_path = os.path.join(args.model_dir, args.model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    core = Core(weights_path,
                args.model_type,
                args.dynamic)

    _, onnx_path, metadata = export_onnx(core,
                                         args.output_dir,
                                         args.input_size,
                                         args.channel,
                                         args.device,
                                         args.half,
                                         args.dynamic,
                                         args.simplify)
    _ = export_trt(onnx_path,
                   metadata,
                   args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        help='must in the version of yolo model ex. v8, v9, v9-pose ...')
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--input_size", type=str, required=True)
    parser.add_argument("--channel", type=int, required=True)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    main(args)