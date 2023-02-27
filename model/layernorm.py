import torch.nn as nn 

class LayerNormalization():
  def __init__(self, embed_size, epsilon=1e-9):
    self.epsilon = epsilon 
    self.gamma = nn.Parameter(torch.ones(embed_size))
    self.beta = nn.Parameter(torch.zeros(embed_size))

  def forward(self, x):
    # mean/std of each layer 
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x-mean)**2).mean(dim=-1, keepdim=True)
    std = (var+self.epsilon).sqrt() # eps: prevent from div to 0 

    y = (x-mean) / std
    out = self.gamma * y + self.beta # make learnable 
    return out 
