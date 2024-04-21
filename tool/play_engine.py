'''
    CUDA_VISIBLE_DEVICES="0" python3 play_engine.py --path ./save/model_* --src ./test/front_view1_1080p_25fps.mp4

'''

import cv2
import torch
from tqdm import tqdm

import argparse

import numpy as np
import time

import tensorrt as trt
from collections import OrderedDict, namedtuple

from utils import non_max_suppression, scale_boxes, LetterBox

class TRTInference():
    def __init__(self, engine_path, device='cpu', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        print(torch.cuda.get_device_name(self.device))
        print(".".join(list(map(str, torch.cuda.get_device_capability(self.device)))))
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.synchronize()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names, self.output_names = self.get_names()
        self.input_shape = self.engine.get_binding_shape(0)

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_names(self):
        input_names, output_names = [], []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        return input_names, output_names
    
    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        '''build binddings'''

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def __call__(self, blob):

        for n in self.input_names:
            if self.bindings[n].shape != blob.shape:
                self.context.set_input_shape(n, blob.shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob.shape)

        self.bindings_addr.update({n: blob.data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        # outputs = {n: self.bindings[n].data for n in self.output_names}
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def preprocessing(frame, device, letterbox):
    
    inputs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inputs = cv2.cvtColor(inputs, cv2.COLOR_GRAY2BGR)

    inputs = letterbox(inputs)                                      # (736, 1280, 3)

    inputs = torch.from_numpy(inputs).to(device)
    inputs = inputs.unsqueeze(0).permute(0,3,1,2).contiguous()      # (1, 3, 736, 1280)
    
    inputs = inputs / 255 
    return inputs


def postprocessing(preds, inp, orig_img):
    """Post-processes predictions and returns a list of Results objects."""

    preds = non_max_suppression(preds, conf_thres=0.01)[0]

    objs = []
    for i, pred in enumerate(preds):
        pred[:4] = scale_boxes(inp.shape[2:], pred[:4], orig_img.shape)   # (x1, y1, x2, y2, conf, cls)
        objs.append(pred)

    return objs

def test(args):

    engine_path = args.path                 # './save/model__tensorrt__trt8.6.1__sm8.6__cuda12.1.engine'
    device = f"cuda:{args.device}"          # cuda:0

    img_test = args.img_test
    video_test = args.video_test
    src = args.src                          # ./test/front_view1_1080p_25fps.mp4
    show = args.imshow

    model_name = args.path.split('/')[-1]   # model__tensorrt__trt8.6.1__sm8.6__cuda12.1.engine
    video_name = args.src.split('/')[-1]    # front_view1_1080p_25fps.mp4
    
    inference = TRTInference(engine_path, device=device, max_batch_size=32, verbose=False)

    input_shape = inference.input_shape[-2:]
    imgsz = (720,1280)
    h,w = imgsz
    letterbox = LetterBox(input_shape)
    
    # image test
    if img_test:
        img = cv2.imread('./test/test.png')                                 # raw (901, 1604, 3)
        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)          # scaling (720, 1280, 3)

        inputs = preprocessing(img, device, letterbox)                      # (1, 3, 7xx, 1280)
        outputs = inference(inputs)['output0']                              # (1, 5, 7xx*1280)
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
                    outputs = inference(inputs)['output0']

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
