import os

import cv2
import numpy as np
import onnxruntime as ort
import torch


def main():

    # test set
    # input_size: (11,3,256,128) * 2
    team_a_dir = '/workspace/resnet/test/PX_PH_A/0'
    team_b_dir = '/workspace/resnet/test/PX_PH_A/1'

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
    inputs = np.array(inputs)                # (22,3,256,128)
    inputs = inputs.astype(np.float16)

    # model
    onnx_path = '/workspace/weights/trt10.0.1__sm8.6__cuda12.4/resnet18_NMI_08107_ARI_08820_F_09410.pt.onnx'
    session = ort.InferenceSession(onnx_path)

    for input in session.get_inputs():
        print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
    for output in session.get_outputs():
        print(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})

    print(inputs.shape)
    print(len(outputs), outputs[0].shape)

    # teamclassifier
    from resnet.teamclassification import affi
    clustered = affi(outputs[0])

    print(clustered)


if __name__=='__main__':
    main()