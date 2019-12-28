import os
import datetime

def get_file_name(dir: str, has_expand: bool=False) -> str:
    file_name = os.path.basename(dir)
    if has_expand:
        return file_name
    return os.path.splitext(file_name)[0]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
