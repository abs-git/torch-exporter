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

logging.basicConfig(level=logging.INFO)
py_logger = logging.getLogger(__name__)

from common.utils import check_imgsz
from resnet.tools.model import get_model

ROOT = os.getcwd()

def export_onnx(model_dir,
                model_name,
                input_size,
                channel,
                output_dir,
                device,
                half,
                dynamic,
                simplify,
                verbose=True):

    inpsz = check_imgsz(input_size)

    model = get_model()
    weights_path = os.path.join(model_dir, model_name)
    checkpint = torch.load(weights_path, map_location=f'cuda:{device}', weights_only=True)
    model.load_state_dict(checkpint)

    onnx_path = os.path.join(output_dir, f'{model_name}.onnx')

    max_batch_size = 100
    dummy = torch.randint(0, 255, (max_batch_size, channel, *inpsz), dtype=torch.float)   # (1,3,256,128)

    model = model.to(f'cuda:{device}')
    dummy = dummy.to(f'cuda:{device}')

    y = None
    for _ in range(3):
        y = model(dummy)  # dry runs
        py_logger.info(f'Dry run: outputs {y.shape}')

    if half:
        model = model.half()
        dummy = dummy.half()
    else:
        model = model.float()
        dummy = dummy.float()

    input_names = ['inputs']
    output_names = ['feats']

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {input_names[0]: {0: 'batch'},
                        output_names[0]: {0: 'batch'}}

    metadata = {'date': datetime.datetime.now().isoformat(),
                'input_names': input_names,
                'output_names': output_names,
                'input_shapes': {'inputs': dummy.shape},
                'max_batch_size': max_batch_size,
                'dynamic': dynamic,
                'model_name': model_name,
                'half': half}

    torch.onnx.export(model.cpu() if dynamic else model,
                      dummy.cpu() if dynamic else dummy,
                      onnx_path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      do_constant_folding=True,
                      opset_version=17,
                      verbose=verbose
                      )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if simplify:
        py_logger.info(f'Use simplify')
        input_shapes = {input_names[0]:dummy.shape}

        onnx_model = onnxslim.slim(onnx_model)
        simplify_model, _ = onnxsim.simplify(onnx_path, test_input_shapes=input_shapes)
        inferred_model = onnx.shape_inference.infer_shapes(simplify_model)
        onnx.save(inferred_model, onnx_path)

    py_logger.info(f'Onnx done..')

    return onnx_model, onnx_path, metadata

def export_trt(onnx_path, metadata, output_dir, rm_onnx=True, verbose=True):

    shapes = metadata['input_shapes']
    dynamic = metadata['dynamic']
    model_name = metadata['model_name']
    half = metadata['half']

    engine_path = os.path.join(output_dir, f'{model_name}.engine')

    logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<30)

    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    # elif int8:
    #     pass

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

    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            print(inp.name, (1, *shapes[inp.name][1:]), (max(1, shapes[inp.name][0]//2), *shapes[inp.name][1:]), shapes[inp.name])
            profile.set_shape(inp.name,
                              (1, *shapes[inp.name][1:]),
                              (max(1, shapes[inp.name][0]//2), *shapes[inp.name][1:]),
                              shapes[inp.name])
        config.add_optimization_profile(profile)


    # engine = builder.build_engine(network, config)                  # tensorrt 8.x
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

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    _, onnx_path, metadata = export_onnx(args.model_dir,
                                         args.model_name,
                                         args.input_size,
                                         args.channel,
                                         output_dir,
                                         args.device,
                                         args.half,
                                         args.dynamic,
                                         args.simplify,
                                         verbose=False)

    _ = export_trt(onnx_path,
                    metadata,
                    output_dir,
                    verbose=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
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