import torch.nn as nn


class PositionalEncoding(nn.Module):
  def __init__(self,  max_len, embed_size, dropout=0.1):
    super().__init__()

    self.embed_size = embed_size
    self.dropout = dropout
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.pe = torch.zeros(max_len, embed_size).to(self.device)
    self.pe.requires_grad=False #

    pos = torch.arange(0, max_len).unsqueeze(1).float()
    
    two_i = torch.arange(0, embed_size, step=2).float()
    div_term = torch.pow(10000, (two_i/torch.Tensor([embed_size]))).float()
    
    self.pe[:, 0::2] = torch.sin(pos/div_term) # even = sin 
    self.pe[:, 1::2] = torch.cos(pos/div_term) # odd = cos 

    # self.pe = self.pe.unsqueeze(0) # [batch, max_len, embed_size]


  # x input from Embedding 
  def forward(self, x): # x = src: src_sent from Transformer -> embedding 
    batch_size, seq_len = x.shape # 

  
    return self.pe[:seq_len, :]
