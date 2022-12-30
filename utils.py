import os
import time
import contextlib
from  pathlib import Path

import cv2
import torch


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def imwrite(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

        

class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()
