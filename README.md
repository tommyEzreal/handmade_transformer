# handmade_transformer
> middle_level implementation   
> low_level implementation

> ⚠️ all codes will be written by hand line-by-line 


```python
# from https://www.youtube.com/watch?v=U0s0f995w14

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # =(embed_size, embed_size)

  def forward(self, values, keys, query, mask):
    N = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # split embedding into self.heads pieces

    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = query.reshape(N, query_len, self.heads, self.head_dim)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)


    """
    # queries shape: (N, query_len, heads, heads_dim)
    # keys shape: (N, key_len, heads, heads_dim)
    # energy shape : (N, heads, query_len, key_len)
    """
    # einsum : instead of matmul 
    energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

    if mask is not None: # masking 
      energy = energy.masked_fill(mask == 0, float("-1e20"))
    
    """
    attention shape : (N, heads, query_len, key_len)
    values shape : (N, value_len, heads, heads_dim)
    after einsum (N, query, heads, head_dim), flatten last two dim 
    """
    #attention = torch.softmax(energy / self.embed_size ** (1/2), dim=3) WRONG?!
    attention = torch.softmax(energy / self.head_dim ** (1/2), dim=3)
    out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
        N, query_len, self.heads * self.head_dim
    )# concat

    out = self.fc_out(out)
    return out 
```

> not using einsum() - try to fix

```python
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, heads, dropout=0.1): 
    # embed_size = d_model, heads = num_heads , head_dim = d_k
    super(MultiHeadAttention,self).__init__()
    self.embed_size = embed_size # d_model  (임베딩차원 , hidden_dim)
    self.heads = heads # num_heads
    self.head_dim = embed_size // heads # dim of each head 

    self.fc_queries = nn.Linear(embed_size, embed_size ,bias=False) # fc layer of Query
    self.fc_keys = nn.Linear(embed_size, embed_size, bias = False) # fc Key
    self.fc_values = nn.Linear(embed_size, embed_size, bias = False) # fc Value

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
    value = self._fc_values(value).view(N, -1, self.heads, self.head_dim)

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
