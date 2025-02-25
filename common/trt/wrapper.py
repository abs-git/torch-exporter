import time
from collections import OrderedDict, namedtuple
from ctypes import c_char_p, cdll
from typing import List, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import torch.backends.cudnn as cudnn

from .allocator import Allocator


class TRTWrapper:
    def __init__(self,
                 engine_path,
                 input_shapes: List[Union[Tuple, List]],
                 device_id: Union[int, str] = 0,
                 verbose=False):
        cudnn.benchmark = True
        torch.set_grad_enabled(False)

        self.engine_path = engine_path
        self.input_shapes = input_shapes
        self.device_id = device_id
        self.device = f'cuda:{device_id}'
        self.verbose = verbose

        self.half = False
        self.dynamic = False
        self.bs = None

        self.__cudaSetDevice()
        self.__initEngine()
        self.__initBindings()
        self.warmup()


    def __initEngine(self):
        logger = trt.Logger(trt.Logger.WARNING if not self.verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")

        with open(self.engine_path, mode="rb") as f, trt.Runtime(logger) as runtime:
            runtime.gpu_allocator = Allocator(self.device)
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            # for name in engine:
            #     profiles = engine.get_tensor_profile_shape(name, 0)
            #     print(profiles)

        self.engine = engine
        self.context = context

    def __initBindings(self):

        Binding = namedtuple("Binding",
                             ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()
        input_names, output_names = [], []

        max_batch_size = -1
        max_channel = -1
        for name in self.engine:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if dtype == np.float16:
                self.half = True

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if -1 in shape:
                    profiles = self.engine.get_tensor_profile_shape(name, 0)
                    batch_sizes = [p[0] for p in profiles]
                    channels = [p[1] for p in profiles]
                    profile_shapes = [p[2:] for p in profiles]

                    for s in self.input_shapes:
                        if tuple(s) in profile_shapes:
                            shape = s

                    if max_batch_size < max(batch_sizes):
                        max_batch_size = max(batch_sizes)

                    if max_channel < max(channels):
                        max_channel = max(channels)

                    shape = (max_batch_size, max_channel, *shape)

                self.context.set_input_shape(name, shape)
                input_names.append(name)

            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                if -1 in shape:
                    shape = (max_batch_size, *shape[1:])

                output_names.append(name)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

        self.input_name = list(sorted(input_names))
        self.output_names = list(sorted(output_names))
        self.bindings = bindings
        self.binding_addrs = binding_addrs
        self.max_batch_size = max_batch_size


    def warmup(self):
        dummy = {n: self.bindings[n].data.clone()
                 for n in self.input_name}
        for _ in range(10):
            t = time.perf_counter()
            self.__call__(dummy)
            torch.cuda.synchronize()
            elapsed_ms = 1e3 * (time.perf_counter() - t)
            fps = 1e3 / elapsed_ms
            if self.verbose:
                # LOGGER.info(
                #     f"Warmup iteration took {elapsed_ms:.2f}ms (fps={fps:.1f})",
                # )
                print(f"Warmup iteration took {elapsed_ms:.2f}ms (fps={fps:.1f})")


    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs dict(torch.Tensor)
                Example:
                    {
                        'input_name1': torch.Tensor,
                        'input_name2': torch.Tensor
                    }

        Returns:
            outputs (dict):
                Example:
                    {
                        'output_name1': torch.Tensor,
                        'output_name2': torch.Tensor,
                        'output_name3': torch.Tensor,
                        ...
                    }
        """

        for name, input in inputs.items():
            input = input.half() if self.half else input.float()

            assert input.shape[1:] == self.bindings[name].shape[1:]

            if input.shape[0] != self.bindings[name].shape[0]:
                self.bindings[name] = self.bindings[name]._replace(shape=input.shape)
                self.bindings[name].data.resize_(input.shape)
                self.context.set_input_shape(name, input.shape)
                self.bs = input.shape[0]

            self.binding_addrs[name] = int(input.data_ptr())

        if self.bs != None:
            for name in self.output_names:
                new_shape = (self.bs, *self.bindings[name].shape[1:])
                self.bindings[name] = self.bindings[name]._replace(shape=new_shape)
                self.bindings[name].data.resize_(new_shape)
            self.bs = None

        self.context.execute_v2(list(self.binding_addrs.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs


    def __cudaSetDevice(self):
        libcudart = cdll.LoadLibrary("libcudart.so")
        libcudart.cudaGetErrorString.restype = c_char_p

        ret = libcudart.cudaSetDevice(int(self.device_id))
        if ret != 0:
            error_string = libcudart.cudaGetErrorString(ret).decode('utf-8')
            raise RuntimeError("cudaSetDevice: " + error_string)


    # TODO: Implement this method
    def set_profiler(self, profiler):
        pass
