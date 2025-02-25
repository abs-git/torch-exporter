import cv2
import numpy as np
import torch
import torchvision

from common.trt.wrapper import TRTWrapper
from common.utils import letterbox, scale_boxes


def main():

    det_img_path = '/workspace/mm/resources/beatles.jpg'
    det_path = '/workspace/weights/model-det__trt10.0.1__sm8.6__cuda12.4/end2end.engine'

    img = cv2.imread(det_img_path)
    x = torch.tensor(img.copy()).permute(2,0,1)
    x, _, _ = letterbox(x, (1344, 1344))
    x = x.unsqueeze(0)
    x = x / 255.0
    x = x.to('cuda:0')

    det = TRTWrapper(engine_path=det_path,
                     input_shapes=[(1344,1344)])
    det.warmup()
    objs = det({'input':x})

    bboxes = objs["bboxes"][0]
    scores = objs["scores"][0]

    keep = scores > 0.2
    keep = keep.squeeze()

    if not torch.any(keep):
        return False

    bboxes = bboxes[keep, :]
    scores = scores[keep, :]

    keep_nms_idx = torchvision.ops.boxes.nms(
        bboxes, scores.squeeze(), 0.5
    )

    bboxes = bboxes[keep_nms_idx, :]
    scores = scores[keep_nms_idx, :]

    bboxes = scale_boxes(x.shape[2:], bboxes, img.shape)

    print(bboxes)

    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (225,0,255), 2)

    cv2.imwrite('/workspace/mm/resources/beatles_det.jpg', img)

if __name__=='__main__':
    main()