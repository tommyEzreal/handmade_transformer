from model.positional_encoding import PositionalEncoding

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout):
    super(TransformerBlock, self).__init__()

    self.attention = MultiHeadAttention(embed_size, heads)
    self.attn_layer_norm = nn.LayerNorm(embed_size)
    self.ff_layer_norm = nn.LayerNorm(embed_size)
    self.feed_forward = FeedForward(embed_size,forward_expansion,dropout)
    self.dropout = nn.Dropout(dropout)
  
  # 하나의 src 임베딩을 Q,K,V로 복제하여 입력 
  def forward(self, src, src_mask):
    # attention block & attn_layer_norm
    _src = self.attention(query = src, key = src, value = src, mask = src_mask)
    src = self.attn_layer_norm(src + self.dropout(_src)) # src 그대로와 attention 통과한 _src 둘다 layer_norm 통과 
    
    # feed_forward block & ff_layer_norm
    _src =self.feed_forward(src)
    src = self.ff_layer_norm(src + self.dropout(_src))    
    
    return src
  
  
# apply position_encoding 

class Encoder(nn.Module):
  def __init__(self,
             src_vocab_size, # input_dim (하나의 단어에 대한 원핫 인코딩의 차원)
             embed_size, # hidden_dim (하나의 단어에 대한 임베딩 차원)
             num_layers, # num_encoder_layer 
             heads, # num_heads 
             forward_expansion, #feedforward dim / embed_size
             dropout,
             device,
             max_length):
    super(Encoder, self).__init__()
    
    self.device = device

    self.embed_size = embed_size

    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.position_embedding = PositionalEncoding(max_length, embed_size)

    self.layers = nn.ModuleList(
        [
            TransformerBlock(
                embed_size,
                heads, 
                forward_expansion = forward_expansion,
                dropout = dropout
            )
        for _ in range(num_layers)] # encoder layer 수만큼 반복 
    ) 
    self.dropout = nn.Dropout(dropout)

  def forward(self, src , src_mask): # src: src_sent from Transformer. 
    # src, src_mask: [batch, src_len]

    # out = src embedding + position encoding 
    
    # word_embedding: [batch, src_len, embed_size]
    # position_embedding: [src_len, embed_size]
    # for each batch, same positional encoding will be applied 
    out = self.dropout(self.word_embedding(src) + self.position_embedding(src))
    
    # 각 encoder layer마다 수행  
    for layer in self.layers:
      out = layer(out, src_mask) # 각 layer의 out이 다시 다음 Layer의 input(src)으로
    # out: [batch, src_len, embed_size]

    return out # 마지막 encoder layer의 출력 
