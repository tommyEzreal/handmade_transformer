import torch
import torch.nn as nn

from model.positional_encoding import PositionalEncoding
from model.multi_head_attention import MultiHeadAttention
from model.feedforward import FeedForward

class DecoderBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout, device):
    super(DecoderBlock, self).__init__()

    # 중략해서도 가능함 
    # self.attention = MultiHeadAttention(embed_size, heads)
    # self.norm = nn.LayerNorm(embed_size)
    # self.transformer_block = TransformerBlock(
    #     embed_size, heads, dropout, forward_expansion
    # )
    # self.dropout = nn.Dropout(dropout)

    self.self_attention = MultiHeadAttention(embed_size, heads)
    self.encoder_attention = MultiHeadAttention(embed_size, heads)

    self.ff_layer_norm = nn.LayerNorm(embed_size)
    self.attn_layer_norm = nn.LayerNorm(embed_size)
    self.enc_layer_norm = nn.LayerNorm(embed_size)

    self.feed_forward = FeedForward(embed_size, forward_expansion, dropout)

    self.dropout = nn.Dropout(dropout)

  def forward(self, trg, enc_out, trg_mask, src_mask):
    
    # trg: [batch, trg_len, embed_size]
    # enc_out: [batch, src_len, embed_size]
    # trg_mask : [batch, trg_len]
    # src_mask : [batch, src_len]

    # self attention block
    _trg = self.self_attention(trg,trg,trg,trg_mask)
    
    # LayerNorm (_trg + trg둘다 입력으로) 
    trg = self.attn_layer_norm(trg+self.dropout(_trg))
    # trg : [batch, trg_len, embed_size]

    # encoder attention block
    # Query만 decoder입력인 trg로 받아오고 Key, Value를 encoder의 마지막 layer로 부터 받은 enc_out 
    _trg = self.encoder_attention(trg, enc_out, enc_out, src_mask)

    # LayerNorm()
    trg = self.enc_layer_norm(trg + self.dropout(_trg))
    # trg size same 

    # feed_forward block
    _trg = self.feed_forward(trg)

    # LayerNorm()
    trg = self.ff_layer_norm(trg + self.dropout(_trg))
    # trg size same 

    return trg


  
# apply positional encoding 

class Decoder(nn.Module):
  def __init__(self,
               trg_vocab_size,
               embed_size,
               num_layers,
               heads,
               forward_expansion,
               dropout,
               device,
               max_length):
    super(Decoder, self).__init__()

    self.device = device
    self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
    self.position_embedding = PositionalEncoding(max_length, embed_size)
    
    self.layers = nn.ModuleList(
        [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
        for _ in range(num_layers)]
    )
    self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, trg, enc_out, trg_mask, src_mask):
    # trg: [batch, trg_len]
    # enc_out: [batch, src_len, embed_size]
    # trg_mask: [batch, trg_len]
    # src_mask: [batch, src_len]


    trg = self.dropout((self.word_embedding(trg))+ self.position_embedding(trg))
    # trg: [batch, trg_len, embed_size]

    for layer in self.layers:
      trg = layer(trg, enc_out, trg_mask, src_mask) # src, trg mask 모두 사용 
    
    out = self.fc_out(trg)
    return out # out : [batch, trg_len, out_dim]
