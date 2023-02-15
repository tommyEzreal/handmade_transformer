# handmade_transformer 🤖
   
> low_level study

> ⚠️ my own implementation of transformer architecture 

<br><br>
## Multi-Head Attention
<img src = "https://user-images.githubusercontent.com/100064247/218194035-e14245fb-f10d-43ab-8f80-88cb0c6dd78d.png" width=50% height=50% align="center" >


```python
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, heads, dropout=0.1): 
    # embed_size = d_model, heads = num_heads , head_dim = d_k
    super(MultiHeadAttention,self).__init__()

    self.embed_size = embed_size # d_model  (임베딩차원 , hidden_dim)
    self.heads = heads # num_heads
    self.head_dim = embed_size // heads # dim of each head 
    assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

    self.fc_queries = nn.Linear(embed_size, embed_size) # fc layer of Query
    self.fc_keys = nn.Linear(embed_size, embed_size) # fc Key
    self.fc_values = nn.Linear(embed_size, embed_size) # fc Value

    self.dropout = nn.Dropout(dropout)
    self.fc_out = nn.Linear(embed_size, embed_size)

  def forward(self, query, key, value, mask=None):
    N = query.shape[0] # N = batch
      
    
    # query = [batch_size, query_len, embed_size]
    # key = [batch_size, key_len, embed_size]
    # value = [batch_size, value_len, embed_size]
    

    # head개수만큼 나눠주기 (num_heads만큼의 K,Q,V)
    # view: [batch, qkv_len, embed_size] -> [batch, qkvlen, n_heads, head_dim]
    query = self.fc_queries(query).view(N , -1, self.heads, self.head_dim)
    key = self.fc_keys(key).view(N, -1, self.heads, self.head_dim)
    value = self.fc_values(value).view(N, -1, self.heads, self.head_dim)

    # = permute(0,2,1,3) = (batch(N), num_heads(self.heads), qkv_len, self.head_dim)
    query = query.transpose(1,2) # [batch, n_heads, query_len, head_dim]
    key = key.transpose(1,2) # [batch, n_heads, key_len, head_dim]
    value = value.transpose(1,2) # [batch, n_heads, value_len, head_dim]


    # dot_product attention calculate
    scaled_dp_attn = self.dot_product_attention(query, key, value, self.head_dim, mask, self.dropout)
    # scaled_dot_product_attention : [batch, n_heads, q_len, head_dim]

    concat = scaled_dp_attn.transpose(1,2).contiguous() # [batch, q_len, n_heads, head_dim]
    concat = concat.view(N, -1, self.embed_size) # 다시 embedsize로 concat [batch, q_len, n_heads*head_dim]
    out = self.fc_out(concat) # fc layer in 

    return out # [batch_size, query_len, embed_size]

  # 내적연산부분 따로 정의 
  def dot_product_attention(self, query, key, value, head_dim, mask=None, dropout=None):
    energy = torch.matmul(query, key.transpose(-2,-1) / math.sqrt(self.head_dim)) # attn energy
    # key.tanspose(-2,-1): [batch, n_heads, key_len, head_dim] -> [batch, n_heads, head_dim, key_len]
    # torch.matmul(q,k.transpose(-2,-1)) : [b,n_h,q_len,h_d] * [b,n_h,h_d,k_len] = [b, n_h, q_len, k_len]
    # /math.sqrt(self.head_dim) = scaling with head_dim  before softmax

    if mask is not None: # mask (optional)
      energy = energy.masked_fill(mask==0, -1e20)

    attention = torch.nn.functional.softmax(energy, dim=-1) # attention score  
    # [batch, n_heads, q_len, k_len]

    if dropout is not None:
      attention = dropout(attention)

    # softamx(QKtrans/scale) dot value
    # softmax( [batch, n_heads, q_len, k_len] * [batch, n_heads, v_len, head_dim] )
    out = torch.matmul(attention, value) # out: [batch, n_heads, q_len, head_dim]
    return out # scaled_dot_product_attention

```

<br><br>
## PositionWise FeedForward
<img width="700" alt="image" src="https://user-images.githubusercontent.com/100064247/218197012-a6128a5a-3575-4766-ad70-2118dcee85e6.png">


```python
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
```


<br><br>
## Encoder
> transformer block = encoder_block 
![image](https://user-images.githubusercontent.com/100064247/218197650-da8be027-5a7f-4e59-a482-cb5e50a45cea.png)



```python
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
```

```python
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
    self.position_embedding = nn.Embedding(max_length, embed_size)

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

  def forward(self, src , src_mask):
    # src, src_mask: [batch, src_len]
    
    N, src_len = src.shape

    positions = torch.arange(0, src_len).expand(N, src_len).to(self.device)
    # positions: [batch, src_len]

    # out = src embedding + position embedding 
    out = self.dropout(self.word_embedding(src) + self.position_embedding(positions))
    
    # 각 encoder layer마다 수행  
    for layer in self.layers:
      out = layer(out, src_mask) # 각 layer의 out이 다시 다음 Layer의 input(src)으로
    # out: [batch, src_len, embed_size]

    return out # 마지막 encoder layer의 출력 
```
<br><br>
## Decoder
![image](https://user-images.githubusercontent.com/100064247/218197876-f5e64f84-71c2-4fb9-b9fa-f5b794dd979a.png)


```python
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
```

```python
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
    self.position_embedding = nn.Embedding(max_length, embed_size)
    
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
    N, trg_len = trg.shape
    
    positions = torch.arange(0, trg_len).expand(N, trg_len).to(self.device)
    # positions: [batch, trg_len]

    trg = self.dropout((self.word_embedding(trg))+ self.position_embedding(positions))
    # trg: [batch, trg_len, embed_size]

    for layer in self.layers:
      trg = layer(trg, enc_out, trg_mask, src_mask) # src, trg mask 모두 사용 
    
    out = self.fc_out(trg)
    return out # out : [batch, trg_len, out_dim]

```
<br><br>
## Transformer
![image](https://user-images.githubusercontent.com/100064247/218198004-870f2079-4819-4691-acb8-10c0db44baec.png)


```python
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
```
## translation task 

### data
#### ko-en sentence pair ( AI Hub 한국어-영어 번역 말뭉치 / 구어체 데이터 1 & 2)
- corpus : 115000
- train: 92000
- valid, test: 11500 
- source_vocab_size : 29004
- target_vocab_size : 19736


### train setting
- 21,554,456 trainable parameters

```python
# model 
embed_size = 256
num_layers = 3
num_heads = 8
dropout = 0.2
max_length = 100

# hyperparameters
epochs = 20
batch_size = 256
learning_rate = 5e-4
optimizer = Adam
```

### train result

```
---------------------------------------------------------
| Epoch: 01  | Train Loss:  5.235  | Valid Loss:  4.350
---------------------------------------------------------
| Epoch: 02  | Train Loss:  4.155  | Valid Loss:  3.833
---------------------------------------------------------
| Epoch: 03  | Train Loss:  3.726  | Valid Loss:  3.499
---------------------------------------------------------
| Epoch: 04  | Train Loss:  3.399  | Valid Loss:  3.272
---------------------------------------------------------
                  . . . . 
| Epoch: 19  | Train Loss:  1.478  | Valid Loss:  2.413
---------------------------------------------------------
| Epoch: 20  | Train Loss:  1.427  | Valid Loss:  2.409
---------------------------------------------------------

---------------------------------------------------------
Test Loss: 2.390
---------------------------------------------------------

```
### bleu score 

```
------------------------------
Total BLEU Score = 15.31
------------------------------
Individual BLEU1 score = 46.40
Individual BLEU2 score = 19.43
Individual BLEU3 score = 10.17
Individual BLEU4 score = 6.00
------------------------------
Cumulative BLEU1 score = 46.40
Cumulative BLEU2 score = 30.03
Cumulative BLEU3 score = 20.93
Cumulative BLEU4 score = 15.31
------------------------------
```

```
[100/1000]
예측: ['the', 'cell', 'phone', 'has', 'a', 'hit', 'and', 'the', 'internet', '.']
정답: ['there', 'is', 'a', 'clock', 'between', 'a', 'cell', 'phone', 'and', 'a', 'scissor', '.']
[200/1000]
예측: ['i', 'do', '<unk>', 'want', 'to', 'meet', 'each', 'other', 'anymore', '.']
정답: ['i', 'do', "n't", 'want', 'to', 'meet', 'you', 'personally', 'anymore', '.']
[300/1000]
예측: ['i', 'think', 'it', 'might', 'be', 'meaningful', 'and', '<unk>', '.']
정답: ['it', 'would', 'be', 'difficult', 'to', 'understand', 'what', 'it', 'means', '.']
[400/1000]
예측: ['practice', 'for', 'a', 'putt', 'before', 'you', 'start', '.']
정답: ['take', 'a', 'few', 'practice', 'swings', 'before', 'hitting', 'a', 'ball', '.']
[500/1000]
예측: ['i', 'need', 'to', 'check', 'the', 'volume', '.']
정답: ['i', 'have', 'to', 'make', 'a', 'call', 'on', 'a', 'landline', '.']
[600/1000]
예측: ['it', 'looks', 'like', 'our', 'team', 'leader', 'asked', 'us', 'to', 'come', '.']
정답: ['it', 'looks', 'like', 'a', 'new', 'bookkeeper', 'will', 'join', 'our', 'team', '.']
[700/1000]
예측: ['technology', 'gives', 'us', 'a', 'lot', 'easier', 'than', 'our', 'computer', '.']
정답: ['computer', 'technology', 'has', 'improved', ' ', 'the', 'way', 'to', 'get', 'information', '.']
[800/1000]
예측: ['i', '<unk>', 'so', 'sorry', ',', 'i', 'did', '<unk>', 'speak', 'wrong', '.']
정답: ['i', "'m", 'awfully', 'sorry', ',', 'it', 'was', 'wrong', 'of', 'me', '.']
[900/1000]
예측: ['the', 'answer', 'may', 'be', 'able', 'to', 'answer', 'the', 'question', '.']
정답: ['the', 'resolution', 'of', 'that', 'question', 'may', 'be', 'difficult', '.']
[1000/1000]
예측: ['so', 'i', 'want', 'to', 'be', 'a', 'great', 'person', 'as', 'possible', '.']
정답: ['so', 'i', 'hope', 'to', 'be', 'an', 'aircrew', 'as', 'soon', 'as', 'possible', '.']
```
## Reference
https://github.com/Huffon/pytorch-transformer-kor-eng    
https://github.com/nawnoes/pytorch-transformer     
https://github.com/hyunwoongko/transformer      
