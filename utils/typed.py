from typing import Union, NamedTuple, Optional
from pathlib import Path
import torch


PathLike = Union[str, Path]


class NamedTensorTuple(NamedTuple):
    def to(self, *args, **kwargs) -> "NamedTensorTuple":
        return self.__class__(
            *(tensor.to(*args, **kwargs) for tensor in self)  # type: ignore
        )

    def cuda(
        self,
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ):
        if device is None:
            device = torch.device(torch.cuda.current_device())
        return self.to(
            device=device, non_blocking=non_blocking, memory_format=memory_format
        )

    def cpu(
        self,
        memory_format: torch.memory_format = torch.preserve_format,
    ):
        return self.to(device="cpu", memory_format=memory_format)

    def float(
        self,
        memory_format: torch.memory_format = torch.preserve_format,
    ):
        return self.to(dtype=torch.float32, memory_format=memory_format)

    def half(
        self,
        memory_format: torch.memory_format = torch.preserve_format,
    ):
        return self.to(dtype=torch.half, memory_format=memory_format)
