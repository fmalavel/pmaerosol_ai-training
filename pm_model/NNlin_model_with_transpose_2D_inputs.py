# https://discuss.pytorch.org/t/how-to-implement-a-neural-network-with-2-d-tensor-as-input-and-output/166787/2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
import numpy as np

class model(nn.Module):

  def __init__(self):
    super(model ,self).__init__()
    layers = []
    hid_in, hid_out = [16, 256, 512, 256], [256, 512, 256, 16]
    #hid_in, hid_out = [3, 256, 512, 256], [256, 512, 256, 1]
    for hin, hout in zip(hid_in, hid_out):
      layers.append(nn.Linear(hin, hout))
    self.layers = nn.ModuleList(layers)

  def forward(self, x: Tensor) -> Tensor:
    for i, layer in enumerate(self.layers):
      print(f"Layer {i}, size: {x.shape}")
      x = layer(x.transpose(-2,-1)).transpose(-2,-1) #transpose to exploit broadcasting over 2d-input
      print(f"Layer {i}, size: {x.shape}")
    return x

x=torch.randn(128,16,2)
#x=torch.randn(2,3,4)

net = model()
net(x)



import torch
from torch.utils.data import random_split

def split(full_dataset, val_percent, test_percent, random_seed=None):
    amount = len(full_dataset)

    test_amount = (
        int(amount * test_percent)
        if test_percent is not None else 0)
    val_amount = (
        int(amount * val_percent)
        if val_percent is not None else 0)
    train_amount = amount - test_amount - val_amount

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        (train_amount, val_amount, test_amount),
        generator=(
            torch.Generator().manual_seed(random_seed)
            if random_seed
            else None))
    
    return train_dataset, val_dataset, test_dataset

full_dataset =  torch.randn(128,16,2).reshape(-1,2) #x[-1]
train_dataset, val_dataset, test_dataset = split(full_dataset, 0.1, 0.1, 42)

print(len(full_dataset)) # length of the dataset
print(len(train_dataset)) # length of the train division
print(len(val_dataset)) # length of the validation division
print(len(test_dataset)) # length of the test division


