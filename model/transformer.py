from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
  def __init__(self,
               src_vocab_size,
               trg_vocab_size,
               src_pad_idx,
               trg_pad_idx,
               embed_size= 256,
               num_layers =3, # in paper,6(encoder, decoder both)
               forward_expansion = 2,
               heads = 8,
               dropout = 0.1,
               device = "cuda",
               max_length = 100): # 128 , 256, 512, .. based on your data
    super(Transformer, self).__init__()


    self.encoder = Encoder(src_vocab_size,
                           embed_size,
                           num_layers,
                           heads,
                           forward_expansion,
                           dropout,
                           device,
                           max_length)

    self.decoder = Decoder(trg_vocab_size,
                           embed_size,
                           num_layers,
                           heads,
                           forward_expansion,
                           dropout,
                           device,
                           max_length)

    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.device = device

  # make_src_mask: 소스문장의 토큰 마스크값을 0으로 
  def make_src_mask(self, src): # src: [batch, src_len]
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # src_mask: [batch,1,1,src_len]
    return src_mask.to(self.device)

  # make_trg_mask: 타겟문장의 각 다음단어를 알 수 없도록 masking (time_step별로)
  def make_trg_mask(self, trg): # trg: [batch, trg_len]
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
    ) 
    return trg_mask.to(self.device)

  def forward(self, src, trg):
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)
    enc_out = self.encoder(src, src_mask) # with src mask 
    out = self.decoder(trg, enc_out, trg_mask, src_mask) # put it into decoder
    return out 
