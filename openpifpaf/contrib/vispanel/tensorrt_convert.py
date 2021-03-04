import torch
from torch2trt import torch2trt
from openpifpaf import decoder, datasets, network
import numpy as np

from . import datamodule

device = torch.device('cuda')

TORCH_EXT = '.pkl'
TRT_EXT = '.pth'

model_name = 'vispanel-noise-epoch80'
model_dir = '/home/xplore/openpifpaf/openpifpaf/contrib/vispanel/models/'

def get_processor():
    model_path = model_dir + model_name + TORCH_EXT
    model_cpu, _ = network.factory(checkpoint=model_path)
    model = model_cpu.to(device)

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(
        head_metas, profile=None, profile_device=device)
    return processor, model

proc, model = get_processor()

# get sample data
x = torch.ones((1, 3,513,513)).cuda()

# convert to TensorRT
model_trt = torch2trt(model, [x])

# check the output against PyTorch
y = model(x)
y_trt = model_trt(x)

print(torch.max(torch.abs(y - y_trt)))

# save the model

torch.save(model_trt.state_dict(), model_dir + model_name + TRT_EXT)
