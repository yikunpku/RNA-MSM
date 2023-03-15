import subprocess
import multiprocessing as mp
from typing import Iterable, List
from functools import partial
import os
import torch


def run_command(gpus, command):
    gpu = gpus.pop()
    print(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, env=env)
    gpus.append(gpu)


def poll_gpu_with_commands(commands: Iterable[List[str]]):
    import os

    device_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device_str:
        try:
            devices = [int(x) for x in device_str.split(",")]  # type: ignore
        except ValueError:
            raise RuntimeError(
                f"Unrecognized setting for CUDA_VISIBLE_DEVICES: {devices}"
            )
    else:
        devices = list(range(torch.cuda.device_count()))

    manager = mp.Manager()
    gpu_list = manager.list(devices)

    run_func = partial(run_command, gpu_list)

    with mp.Pool(len(devices)) as pool:
        pool.map(run_func, commands)
