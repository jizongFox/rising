import functools

import torch
from loguru import logger


class gpu_timeit:

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        elapsed_time = self.start.elapsed_time(self.end)
        logger.opt(depth=1).info(f"operation time: {elapsed_time / 1000:.4f}.")

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)

        return wrapped_func
