import os

import cv2
import torch

from common.trt.wrapper import TRTWrapper
from resnet.tools.model import get_model


def main():

    # test set
    # input_size: (11,3,256,128) * 2
    team_a_dir = '/workspace/resnet/0'
    team_b_dir = '/workspace/resnet1'

    team_a_samples = os.listdir(team_a_dir)[:11]
    team_b_samples = os.listdir(team_b_dir)[:11]

    team_a_tensors = []
    for n in team_a_samples:
        p = os.path.join(team_a_dir, n)
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,256))    # (h,w,c)
        tensor = torch.tensor(img).permute(2,0,1) / 255.0   # (c,h,w)
        team_a_tensors.append(tensor)

    team_b_tensors = []
    for n in team_b_samples:
        p = os.path.join(team_b_dir, n)
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,256))    # (h,w,c)
        tensor = torch.tensor(img).permute(2,0,1) / 255.0   # (c,h,w)
        team_b_tensors.append(tensor)

    inputs = team_a_tensors + team_b_tensors
    inputs = torch.stack(inputs).to('cuda:0')                # (22,3,256,128)

    # model
    engine_path = '/workspace/weights/test.engine'

    engine = TRTWrapper(engine_path=engine_path,
                        input_shapes=[(256,128)],
                        device_id=0)

    outputs = engine({'inputs':inputs})

    print(inputs.shape)
    print(outputs['feats'].shape)

    # teamclassifier
    from resnet.teamclassification import affi
    clustered = affi(outputs['feats'])

    print(clustered)


if __name__=='__main__':
    main()
