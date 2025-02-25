import os

import cv2
import torch

from common.trt.wrapper import TRTWrapper
from common.utils import letterbox, nms_8, scale_boxes


def main():
    img_path = '/workspace/ultra/resources/tennis.jpg'

    img = cv2.imread(img_path)
    x = torch.tensor(img.copy()).permute(2,0,1)
    x, _, _ = letterbox(x, (768,1280))
    x = x.unsqueeze(0)
    x = x / 255.0
    x = x.to('cuda:0')


    engine_dir = '/workspace/weights/tennis_ball_yolov8s-p6_trt10.0.1__sm8.6__cuda12.4/'
    engine_name = 'yolov8s.engine'

    engine_path = os.path.join(engine_dir, engine_name)
    engine = TRTWrapper(engine_path=engine_path,
                       input_shapes=[(768,1280)],
                       device_id=0)
    preds = engine({'input':x})

    dets = nms_8(preds["output"],
                 conf_thres=0.4,
                 iou_thres=0.4,
                 agnostic=False,
                 max_det=1000,
                 nc=1)[0]

    results = []
    if len(dets):
        dets[:, :4] = scale_boxes(x.shape[2:],
                                  dets[:,:4],
                                  img.shape).round()

        for det in reversed(dets):

            xyxy = det[:4]
            conf = det[4].unsqueeze(-1)
            cls = det[5].unsqueeze(-1)
            center = torch.tensor([((xyxy[2] + xyxy[0]) / 2),
                                    ((xyxy[3] + xyxy[1]) / 2)], dtype=xyxy.dtype)

            results.append([xyxy, conf, cls, center])

    for r in results:
        bbox = r[0]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (225,255,0), 2)

    cv2.imwrite('/workspace/ultra/resources/tennis-ball.jpg', img)


if __name__=='__main__':
    main()