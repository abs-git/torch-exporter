from ultralytics import YOLO


class Core(object):
    def __init__(self, path, type, dynamic) -> None:
        self.path = path
        self.model_type = type
        self.dynamic = dynamic

        if 'v8' in self.model_type.lower():
            self.get_yolov8()
        elif 'v9' in self.model_type.lower():
            if 'pose' in self.model_type.lower():
                self.get_yolov9_pose()
            else:
                self.get_yolov9()
        elif 'v10' in self.model_type.lower():
            self.get_yolov10()
        elif 'v11' in self.model_type.lower():
            if 'pose' in self.model_type.lower():
                self.get_yolov11_pose()
            else:
                self.get_yolov11()

    def get_yolov8(self):

        yolov8 = YOLO(self.path)

        model = yolov8.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Detect':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride

    def get_yolov9(self):

        yolov9 = YOLO(self.path)

        model = yolov9.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Detect':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride

    def get_yolov9_pose(self):

        yolov9_pose = YOLO(self.path)

        model = yolov9_pose.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Pose':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride


    def get_yolov10(self):

        yolov10 = YOLO(self.path)

        model = yolov10.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'v10Detect':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride


    def get_yolov11(self):

        yolov11 = YOLO(self.path)

        model = yolov11.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Detect':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride


    def get_yolov11_pose(self):

        yolov11_pose = YOLO(self.path)

        model = yolov11_pose.model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for n, m in model.named_modules():
            if type(m).__name__ == 'Pose':
                m.dynamic = self.dynamic
                m.export = True
                m.format = 'onnx'
            elif type(m).__name__ == 'C2f':
                m.forward = m.forward_split

        self.model = model
        self.stride = model.stride
