import torch
import torch.functional as F
import pandas as pd
import torch.nn as nn

class high_level_target(nn.Module):
    def __init__(self, input_size=3,output_size=1):
        super().__init__()
        self.layer=nn.Sequential(nn.Linear(input_size,256),
                                 nn.SiLU(),
                                 nn.Linear(128,64),
                                 nn.SiLU(),
                                 nn.Linear(64,32),
                                 nn.SiLU()
                                 )
        self.spectlayer=nn.Linear(32,output_size)
        self.memory=[]

    def forward(self,x):
        x=self.layer(x)
        nn.utils.parametrizations.spectral_norm(self.spectlayer,name="weight")
        Value=self.spectlayer(x)
        return Value

class high_level_local(nn.Module):
    def __init__(self, input_size=3,output_size=1):
        super().__init__()
        self.layer=nn.Sequential(nn.Linear(input_size,256),
                                 nn.SiLU(),
                                 nn.Linear(128,64),
                                 nn.SiLU(),
                                 nn.Linear(64,32),
                                 nn.SiLU()
                                 )
        self.spectlayer=nn.Linear(32,output_size)
        self.memory=[]

    def forward(self,x):
        x=self.layer(x)
        nn.utils.parametrizations.spectral_norm(self.spectlayer,name="weight")
        Value=self.spectlayer(x)
        return Value