import torch as T
import torch.functional as F
import pandas as pd
import torch.nn as nn

class low_level_actor(nn.Module):
    def __init__(self):
        self.linear= nn.Sequential()

    def forward(self,x):
