from typing import Any, Optional, Union

import cv2
import torch
import torchvision
from mmdeploy.apis import build_task_processor

from common.utils import letterbox, scale_boxes


def torch2onnx(model_category: str,
               img_path: Any,
               onnx_path: str,
               deploy_cfg: Union[str, dict],
               model_cfg: Union[str, dict],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               verbose = False):

    onnx_cfg = deploy_cfg['onnx_config']
    input_shape = onnx_cfg['input_shape']

    # create model an inputs
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    torch_model = task_processor.build_pytorch_model(model_checkpoint)

    data, model_inputs = task_processor.create_input(
        img_path,
        input_shape,
        data_preprocessor=getattr(torch_model, 'data_preprocessor', None))
    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    data_samples = data['data_samples']

    # onnx config
    opset_version = onnx_cfg.get('opset_version', True)
    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    dynamic_axes = onnx_cfg['dynamic_axes']
    keep_initializers_as_inputs = onnx_cfg.get('keep_initializers_as_inputs', True)

    half = deploy_cfg.backend_config.common_config.fp16_mode

    # model select
    if model_category == 'rtmdet':
        from mm.arch.rtmdet import End2End
        model = End2End(torch_model)

        output_names = ['bboxes', 'scores']
        dynamic_axes = {'input':  {0: 'batch', 2: 'height', 3: 'width'},
                        'bboxes': {0: 'batch'},
                        'scores': {0: 'batch'}}

    elif model_category == 'rtmpose':
        from mm.arch.rtmpose import End2End
        model = End2End(torch_model, data_samples)

        # output_names = ['locs', 'vals']
        # dynamic_axes = {'input': {0: 'batch'},
        #                 'locs':  {0: 'batch'},
        #                 'vals':  {0: 'batch'}}

        output_names = ['simcc_x', 'simcc_y']
        dynamic_axes = {'input': {0: 'batch'},
                        'simcc_x':  {0: 'batch'},
                        'simcc_y':  {0: 'batch'}}

    elif model_category == 'rtmo':
        from mm.arch.rtmo import End2End
        model = End2End(torch_model, data_samples, model_cfg.model)

        # output_names = ['bboxes','scores']
        # dynamic_axes = {'input': {0: 'batch'},
        #                 'bboxes': {0: 'batch'},
        #                 'scores': {0: 'batch'}}

        output_names = ['bboxes','bboxes_scores','keypoints','keypoints_scores']
        dynamic_axes = {'input': {0: 'batch'},
                        'bboxes': {0: 'batch'},
                        'bboxes_scores': {0: 'batch'},
                        'keypoints': {0: 'batch'},
                        'keypoints_scores': {0: 'batch'}}

    else:
        raise ValueError(f"Not supported model: {model_category}")

    model.to(device)
    model_inputs.to(device)

    model = model.float()
    model_inputs = model_inputs.float()

    for _ in range(3):
        # dry run
        y = model(model_inputs)

    torch.onnx.export(
        model,
        model_inputs,
        onnx_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        verbose=verbose)

    check = True
    if check:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

    simplify = True
    if simplify:
        import onnx
        import onnxsim
        input_shapes = {input_names[0]:model_inputs.shape}

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
        output = session.run(None, {input_name: model_inputs.detach().cpu().numpy()})