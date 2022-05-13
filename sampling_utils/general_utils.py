import os

import torch
import torch.nn as nn
from torch.autograd import Variable


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_var(x, device=None):
    """
    x: torch Tensor
    """
    if (torch.cuda.is_available()) and (device is None):
        x = x.cuda()
    elif device is not None:
        x = x.to(device)
    return Variable(x)


def to_np(x):
    """
    x: torch Variable
    """
    return x.data.cpu().numpy()


def init_params_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal(m.weight)
        m.bias.data.zero_()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


def send_file_to_remote(path_to_file, port_to_remote, path_to_save_remote):
    if (port_to_remote is not None) and (path_to_save_remote is not None):
        command = f"scp -P {port_to_remote} ".format(port_to_remote)
        command += path_to_file
        command += " localhost:"
        command += path_to_save_remote
        print(
            f"Try to send file {path_to_file} to remote server....".format(
                path_to_file,
            ),
        )
        os.system(command)
