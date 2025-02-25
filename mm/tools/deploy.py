import argparse
import os

import mmdeploy

from mm.tools.onnx2tensorrt import onnx2trt
from mm.tools.pytorch2onnx import torch2onnx


def main():
    args = parse_args()

    model_category = args.model_category
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    img_path = args.img_path
    work_dir = args.work_dir
    device = args.device

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    prefix = 'end2end'
    onnx_path = os.path.join(work_dir, f'{prefix}.onnx')
    engine_path = os.path.join(work_dir, f'{prefix}.engine')

    deploy_cfg, model_cfg = mmdeploy.utils.load_config(deploy_cfg_path, model_cfg_path)

    input_shapes = deploy_cfg['backend_config']['model_inputs'][0]['input_shapes']

    deploy_cfg['backend_config']['common_config']['fp16_mode'] = True
    fp16_mode = deploy_cfg['backend_config']['common_config']['fp16_mode']

    torch2onnx(model_category=model_category,
               img_path=img_path,
               onnx_path = onnx_path,
               deploy_cfg=deploy_cfg,
               model_cfg=model_cfg,
               model_checkpoint=checkpoint_path,
               device=device)

    onnx2trt(onnx_model=onnx_path,
             engine_path=engine_path,
             input_shapes=input_shapes,
             fp16_mode=fp16_mode,
             max_workspace_size=30,
             int8_mode=False,
             int8_param=None,
             device_id=0,
             verbose=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backends.')
    parser.add_argument('model_category', help='model category ex. rtmpose, rtmdet')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img_path', help='image used to convert model model')
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()