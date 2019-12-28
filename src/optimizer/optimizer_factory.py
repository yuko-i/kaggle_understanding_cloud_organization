import torch
from optimizer.RAdam import RAdam
from optimizer.NAdam import NAdam

def get_optimizer(optimizer_name, params):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(params)
    elif optimizer_name == "RAdam":
        optimizer = RAdam(params)
    elif optimizer_name == "NAdam":
        optimizer = NAdam(params)
    return optimizer