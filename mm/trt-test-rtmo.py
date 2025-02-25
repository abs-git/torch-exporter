import cv2
import numpy as np
import torch
import torchvision

from common.trt.wrapper import TRTWrapper
from common.utils import letterbox, scale_boxes, scale_coords


def main():

    bottomup_img_path = '/workspace/mm/resources/beatles.jpg'
    bottomup_path = '/workspace/weights/rtmo-l__trt10.0.1__sm8.6__cuda12.4/end2end.engine'
    # bottomup_path = '/workspace/weights/rtmo-l__trt10.0.1__sm8.9__cuda12.4/end2end.engine'

    img = cv2.imread(bottomup_img_path)
    x = torch.tensor(img.copy()).permute(2,0,1)
    x = x.unsqueeze(0)
    x, _, _ = letterbox(x, (1344,1344))
    # x = x / 255.0
    x = x.to('cuda:0')

    rtmo = TRTWrapper(engine_path=bottomup_path,
                     input_shapes=[(1344,1344)],
                     device_id=0)
    rtmo.warmup()

    print(rtmo.half)

    objs = rtmo({'input':x})

    bboxes = objs['bboxes'][0]
    bboxes_scores = objs['bboxes_scores'][0]
    kpts = objs['keypoints'][0]
    kpts_scores = objs['keypoints_scores'][0]

    print(bboxes.shape)
    print(bboxes_scores.shape)
    print(kpts.shape)
    print(kpts_scores.shape)

    keep = bboxes_scores > 0.5
    keep = keep.squeeze()

    if not torch.any(keep):
        return False

    bboxes = bboxes[keep, :]
    bboxes_scores = bboxes_scores[keep, :]

    kpts = kpts[keep, :]
    kpts_scores = kpts_scores[keep, :]

    keep_nms_idx = torchvision.ops.boxes.nms(
        bboxes, bboxes_scores.squeeze(), 0.5
    )

    bboxes = bboxes[keep_nms_idx, :]
    bboxes_scores = bboxes_scores[keep_nms_idx, :]

    kpts = kpts[keep_nms_idx, :]
    kpts_scores = kpts_scores[keep_nms_idx, :]

    bboxes = scale_boxes(x.shape[2:], bboxes, img.shape)
    kpts = scale_coords(x.shape[2:], kpts, img.shape)

    print(bboxes.shape)
    print(bboxes_scores.shape)
    print(kpts.shape)
    print(kpts_scores.shape)

    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (225,0,255), 2)

    for i, k in enumerate(kpts):
        for p in k:
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (125,255,125), 1)
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (125,255,150), 1)
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (125,255,175), 1)

    cv2.imwrite('/workspace/mm/resources/beatles_rtmo.jpg', img)

if __name__=='__main__':
    main()