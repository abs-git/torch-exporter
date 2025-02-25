import logging
from typing import Dict, Optional, Sequence, Union

import onnx
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
py_logger = logging.getLogger(__name__)


def onnx2trt(onnx_model: Union[str, onnx.ModelProto],
             engine_path: str,
             input_shapes: Dict[str, Sequence[int]],
             max_workspace_size: int = 30,
             fp16_mode: bool = False,
             int8_mode: bool = False,
             int8_param: Optional[dict] = None,
             device_id: int = 0,
             verbose=False) -> trt.ICudaEngine:

    if int8_mode or device_id != 0:
        import pycuda.autoinit

    if device_id != 0:
        import os
        old_cuda_device = os.environ.get('CUDA_DEVICE', None)
        os.environ['CUDA_DEVICE'] = str(device_id)
        if old_cuda_device is not None:
            os.environ['CUDA_DEVICE'] = old_cuda_device
        else:
            os.environ.pop('CUDA_DEVICE')

    logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    # onnx parsing
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        parse_valid = parser.parse_from_file(onnx_model)
    elif isinstance(onnx_model, onnx.ModelProto):
        parse_valid = parser.parse(onnx_model.SerializeToString())
    else:
        raise TypeError('Unsupported onnx model type!')

    if not parse_valid:
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')


    # config builder
    config = builder.create_builder_config()

    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6<<max_workspace_size)
    else:
        config.max_workspace_size = max_workspace_size

    # dynamic
    profile = builder.create_optimization_profile()
    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        if not getattr(builder, 'platform_has_fast_fp16', True):
            logger.warning('Platform does not has fast native fp16.')
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        if not getattr(builder, 'platform_has_fast_int8', True):
            logger.warning('Platform does not has fast native int8.')
        from .calib_utils import HDF5Calibrator
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5Calibrator(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))

    # create engine
    if hasattr(builder, 'build_serialized_network'):
        engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'


    # i/o
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        py_logger.info(f'input: {inp.name} with shape {inp.shape} {inp.dtype}')
    for out in outputs:
        py_logger.info(f'output: {out.name} with shape {out.shape} {out.dtype}')


    with open(engine_path, mode='wb') as f:
        if isinstance(engine, trt.ICudaEngine):
            engine = engine.serialize()
        f.write(bytearray(engine))