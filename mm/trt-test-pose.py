import cv2
import numpy as np
import torch

from common.trt.wrapper import TRTWrapper
from common.utils import letterbox, scale_coords


def main():

    pose_img_path = '/workspace/mm/resources/human-pose.jpg'
    pose_path = '/workspace/weights/model.engine'

    img = cv2.imread(pose_img_path)
    x = torch.tensor(img.copy()).permute(2,0,1)
    x, _, _ = letterbox(x, (256, 192))
    x = x.unsqueeze(0)
    x = x / 255.0
    x = x.to('cuda:0')

    pose = TRTWrapper(engine_path=pose_path,
                      input_shapes=[(256,192)])
    pose.warmup()

    o = pose({'input':x})

    locs, vals = get_simcc_maximum(o['simcc_x'], o['simcc_y'])
    kpts = scale_coords(x.shape[2:], locs, img.shape)

    print(kpts)

    for k in kpts[0]:
        cv2.circle(img, (int(k[0]), int(k[1])), 1, (225,255,25), 1)
        cv2.circle(img, (int(k[0]), int(k[1])), 3, (225,255,50), 1)
        cv2.circle(img, (int(k[0]), int(k[1])), 5, (225,255,75), 1)

    cv2.imwrite('/workspace/mm/resources/human-pose_keypoints.jpg', img)


def get_simcc_maximum(simcc_x: torch.Tensor,
                      simcc_y: torch.Tensor,
                      simcc_split_ratio: float = 2.0) -> torch.Tensor:
    N, K, _ = simcc_x.shape
    simcc_x = simcc_x.flatten(0, 1)
    simcc_y = simcc_y.flatten(0, 1)
    x_locs = simcc_x.argmax(dim=1, keepdim=True)
    y_locs = simcc_y.argmax(dim=1, keepdim=True)
    locs = torch.cat((x_locs, y_locs), dim=1).to(torch.float32)
    max_val_x, _ = simcc_x.max(dim=1, keepdim=True)
    max_val_y, _ = simcc_y.max(dim=1, keepdim=True)
    vals, _ = torch.cat([max_val_x, max_val_y], dim=1).min(dim=1)
    locs = locs.reshape(N, K, 2)
    locs /= simcc_split_ratio
    vals = vals.reshape(N, K)
    return locs, vals


if __name__=='__main__':
    main()
