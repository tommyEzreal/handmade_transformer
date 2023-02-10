class FeedForward(nn.Module):
  def __init__(self, embed_size, forward_expansion, dropout):
    super(FeedForward,self).__init__()

    self.fc_1 = nn.Linear(embed_size, forward_expansion * embed_size)
    self.fc_2 = nn.Linear(forward_expansion * embed_size, embed_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x): # x: [batch, seq_len, embed_size]
    x = self.fc_1(x) # [batch, seq_len, forward_expansion * embed_size]
    x = self.relu(x) 
    x = self.dropout(x)
    x = self.fc_2(x) # [batch, seq_len, embed_size ]
    return x 
