import tensorrt as trt
import torch


class Allocator(trt.IGpuAllocator):
    def __init__(self,
                 device_id: int = 0,
                 verbose: int = 0):
        super().__init__()

        self.device_id = device_id
        self.verbose = verbose
        self.mems = set()
        self.caching_delete = torch._C._cuda_cudaCachingAllocator_raw_delete
        self.size = None

    def __del__(self):
        mems = self.mems.copy()
        (self.deallocate(mem) for mem in mems)

    def allocate(self: trt.IGpuAllocator,
                 size: int,
                 alignment: int,
                 flags: int) -> int:
        self.size = size

        torch_stream = torch.cuda.current_stream(self.device_id)

        if self.verbose:
            print(f'allocate {size} memory on device {self.device_id} with TorchAllocator.')
        assert alignment >= 0
        if alignment > 0:
            size = size | (alignment - 1) + 1
        mem = torch.cuda.caching_allocator_alloc(
            size, device=self.device_id, stream=torch_stream)
        self.mems.add(mem)
        return mem

    def deallocate(self: trt.IGpuAllocator, memory: int) -> bool:
        if memory not in self.mems:
            return False

        if self.verbose:
            print(f'Delete allocated {self.size} memory on device {self.device_id} without TorchAllocator.')

        self.caching_delete(memory)
        self.mems.discard(memory)
        return True