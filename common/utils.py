import math
from typing import Union


def check_imgsz(imgsz: Union[int, str, list, tuple],
                stride: int = None,
                verbose: bool = False) -> list:

    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    if isinstance(imgsz, str):
        if ',' in imgsz:
            imgsz = list(map(int, imgsz.split(',')))
            imgsz = [imgsz[0], imgsz[1]]
        else:
            imgsz = [int(imgsz), int(imgsz)]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = [int(imgsz[0]), int(imgsz[1])]

    if stride != None:
        min_dim, max_dim, floor = 2, 4, 0
        sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
        if sz != imgsz and verbose == True:
            print(f"WARNING imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")
        if min_dim == 2 and len(sz) == 1:
            sz = [sz[0], sz[0]]
        elif min_dim == 1 and len(sz) == 1:
            sz = sz[0]
        else:
            sz = sz
        return sz

    return imgsz


import torchvision
from torchvision.transforms import transforms


def letterbox(inputs,
              new_shape=(640, 640),
              color=114):
    shape = inputs.shape[-2:]  # (height, width)

    if isinstance(new_shape, int):                  # 640
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, str):                # '640,640'
        new_shape = map(int, new_shape.split(','))
        new_shape = (new_shape[0], new_shape[1])
    elif isinstance(new_shape, list):               # [640,640]
        new_shape = (new_shape[0], new_shape[1])

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw = (new_shape[1] - new_unpad[0])/2
    dh = (new_shape[0] - new_unpad[1])/2

    if shape[::-1] != new_unpad:
        inputs = transforms.Resize(
            (new_unpad[1], new_unpad[0])).forward(inputs)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    inputs = torchvision.transforms.functional.pad(inputs,
                                                   [left, top, right, bottom],
                                                   fill=color,
                                                   padding_mode="constant")

    return inputs, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    boxes[:, 0].clamp_(0, shape[1])
    boxes[:, 1].clamp_(0, shape[0])
    boxes[:, 2].clamp_(0, shape[1])
    boxes[:, 3].clamp_(0, shape[0])


def clip_coords(coords, shape):
    coords[..., 0].clamp_(0, shape[1])
    coords[..., 1].clamp_(0, shape[0])


def scale_boxes(input_shape, boxes, image_shape):
    gain = min(input_shape[0] / image_shape[0],
               input_shape[1] / image_shape[1])
    pad = round((input_shape[1] - image_shape[1] * gain) / 2 -
                0.1), round((input_shape[0] - image_shape[0] * gain) / 2 - 0.1)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    clip_boxes(boxes, image_shape)
    return boxes


def scale_coords(input_shape, coords, image_shape):
    gain = min(input_shape[0] / image_shape[0],
               input_shape[1] / image_shape[1])
    pad = (input_shape[1] - image_shape[1] * gain) / 2, (
        input_shape[0] - image_shape[0] * gain) / 2
    coords[..., 0] -= pad[0]
    coords[..., 1] -= pad[1]
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, image_shape)
    return coords


# yolo

import time

import torch


def xywh2xyxy_8(x):
    y = torch.empty_like(x) if isinstance(x,
                                          torch.Tensor) else np.empty_like(x)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y

def nms_8(preds, conf_thres, iou_thres, agnostic, max_det, classes=None, nc=0):
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1

    bs = preds.shape[0]
    nc = nc or (preds.shape[1] - 4)
    nm = preds.shape[1] - nc - 4
    mi = 4 + nc
    xc = preds[:, 4:mi].amax(1) > conf_thres

    time_limit = 0.5 + 0.05 * bs
    max_wh = 7680
    max_nms = 30000

    preds = preds.transpose(-1, -2)
    preds[..., :4] = xywh2xyxy_8(preds[..., :4])

    tick = time.perf_counter()
    output = [torch.zeros(0, 6 + nm, device=preds.device)] * bs

    for xi, x in enumerate(preds):
        x = x[xc[xi]]

        # skip if none remain
        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask),
                      1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        num_box = x.shape[0]

        # if none remain process next image
        if not num_box:
            continue
        # sort by confidence if exceeds max_nms
        elif num_box > max_nms:
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # batched nms
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes = x[:, :4] + c  # boxes (offset by class)
        scores = x[:, 4]

        keep_box_idx = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        if keep_box_idx.shape[0] > max_det:
            keep_box_idx = keep_box_idx[:max_det]

        output[xi] = x[keep_box_idx]

        if (time.perf_counter() - tick) > time_limit:
            print("NMS time limit exceeded")
            break

    return output

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(preds, conf_thres, iou_thres, agnostic, max_det, classes=None):
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1

    bs = preds.shape[0]
    xc = preds[..., 4] > conf_thres

    max_wh = 4096
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs

    tick = time.perf_counter()

    output = [torch.zeros(0, 6, device=preds.device)] * bs

    for xi, x in enumerate(preds):
        x = x[xc[xi]]

        # skip if none remain
        if not x.shape[0]:
            continue

        # conf = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # x, y, w, h -> x1, y1, x2, y2
        box = xywh2xyxy(x[:, :4])

        # only keep the best class
        conf, class_idx = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, class_idx.float()),
                      1)[conf.view(-1) > conf_thres]

        # filter by classes
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        num_box = x.shape[0]

        # if none remain process next image
        if not num_box:
            continue
        # sort by confidence if exceeds max_nms
        elif num_box > max_nms:
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # batched nms
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes = x[:, :4] + c  # boxes (offset by class)
        scores = x[:, 4]

        keep_box_idx = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        if keep_box_idx.shape[0] > max_det:
            keep_box_idx = keep_box_idx[:max_det]

        output[xi] = x[keep_box_idx]

        if (time.perf_counter() - tick) > time_limit:
            print("NMS time limit exceeded")
            break

    return output
