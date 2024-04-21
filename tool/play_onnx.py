'''
    CUDA_VISIBLE_DEVICES="0" python3 play_engine.py --path ./save/model_* --src ./test/front_view1_1080p_25fps.mp4

'''

import os
import time
import copy

import cv2
import numpy as np
import torch
import onnxruntime

from utils import non_max_suppression, scale_boxes, LetterBox
from ultralytics.utils.checks import check_imgsz

class ONNXInference():
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
    
        self.session = self.get_session()
        self.input_name, self.output_names = self.get_names()
    
    def get_session(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)
        return session
    
    def get_names(self):
        input_name = self.session.get_inputs()[0].name
        output_names = self.session.get_outputs()[0].name
        return input_name, output_names

    def __call__(self, blob):
        '''output : (1,1,5,1280*1280)
        '''
        return self.session.run([self.output_names], {self.input_name: blob})


def preprocessing(frame, imgsz, letterbox):

    inputs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inputs = cv2.cvtColor(inputs, cv2.COLOR_GRAY2BGR)

    inputs = letterbox(inputs)                      # (736, 1280, 3)

    inputs = np.expand_dims(inputs, axis=0)         # (1, 736, 1280, 3)
    inputs = inputs.transpose((0,3,1,2))            # (1, 3, 736, 1280)
    inputs = np.ascontiguousarray(inputs)
    
    inputs = inputs / 255                           # norm
    inputs = inputs.astype(np.float32)

    return inputs


def postprocessing(preds, inp, orig_img):
    """Post-processes predictions and returns a list of Results objects."""

    preds = torch.from_numpy(preds).to('cuda')
    preds = non_max_suppression(preds, conf_thres=0.25)

    objs = []
    for i, pred in enumerate(preds):
        pred[:, :4] = scale_boxes(inp.shape[2:], pred[:, :4], orig_img.shape)   # (x1, y1, x2, y2, conf, cls)
        objs.append(pred[0])

    return objs


def test():
    
    onnx_path = args.path                 # './save/model__tensorrt__trt8.6.1__sm8.6__cuda12.1.onnx'
    device = f"cuda:{args.device}"          # cuda:0

    img_test = args.img_test
    video_test = args.video_test
    src = args.src                          # ./test/front_view1_1080p_25fps.mp4
    show = args.imshow

    model_name = args.path.split('/')[-1]   # model__tensorrt__trt8.6.1__sm8.6__cuda12.1.onnx
    video_name = args.src.split('/')[-1]    # front_view1_1080p_25fps.mp4
    
    inference = TRTInference(onnx_path, device=device, max_batch_size=32, verbose=False)

    input_shape = inference.input_shape[-2:]
    imgsz = (720,1280)
    h,w = imgsz
    letterbox = LetterBox(input_shape)

    # image test
    if img_test:
        img = cv2.imread('./test/test.png')                                 # raw (901, 1604, 3)
        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)          # scaling (720, 1280, 3)

        inputs = preprocessing(img, device, letterbox)                      # (1, 3, 7xx, 1280)
        outputs = inference(inputs)[0]                                       # (1, 5, 7xx*1280)
        objs = postprocessing(outputs, inputs, img)

        for b in objs:
            x1, y1, x2, y2, conf, clss = b
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (225,0,255), 2)
            cv2.putText(img, str(round(conf.item(),2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
        cv2.imwrite(f'{engine_path}.jpg', img)

    # video test
    if video_test:

        cap = cv2.VideoCapture(src)
      
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'./save/test_{model_name}_{video_name}', fourcc, fps, (w,h))

        with tqdm(total=frame_count, desc='saving video') as pbar:

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (w,h), interpolation=cv2.INTER_AREA)      # scaling
                try:

                    t1 = time.time()
                    inputs = preprocessing(frame, device, letterbox)

                    t2 = time.time()
                    outputs = inference(inputs)[0]

                    t3 = time.time()
                    objs = postprocessing(outputs, inputs, frame)

                    t4 = time.time()

                    descript = [f"runtime : {round((t4-t1)*1000,1)}ms  ",
                                f"preprocessing : {round((t2-t1)*1000,1)}ms  ",
                                f"inference : {round((t3-t2)*1000,1)}ms  ",
                                f"postprocessing : {round((t4-t3)*1000,1)}ms  "
                                ]

                    best = False
                    if best:
                        box = sorted(objs, key=lambda x:-x[4])[0]
                        x1, y1, x2, y2, conf, clss = box
                        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (225,0,255), 2)
                        cv2.putText(frame, str(round(conf.item(),2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                    else:
                        for b in objs:
                            x1, y1, x2, y2, conf, clss = b
                            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (225,0,255), 2)
                            cv2.putText(frame, str(round(conf.item(),2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

                    for i, l in enumerate(descript):
                        x = 10
                        y = 10 + i*30
                        cv2.putText(frame, l, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                except:
                    pass
                
                out.write(frame)

                if show:
                    cv2.imshow('title', frame)
                
                if cv2.waitKey(20) == 27:
                    break

                pbar.update(1)
                pbar.set_description(f'{"".join(descript)}')

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--img_test", type=bool, default=False)
    parser.add_argument("--video_test", type=bool, default=True)
    parser.add_argument("--imshow", type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()

    test(args)
