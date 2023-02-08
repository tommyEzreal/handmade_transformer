# handmade_transformer
> middle_level implementation   
> low_level implementation

> ⚠️ all codes will be written by hand line-by-line 


```python
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

> not using einsum()    

```python
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, heads, dropout=0.1): 
    # embed_size = d_model, heads = num_heads , head_dim = d_k
    super(MultiHeadAttention,self).__init__()
    self.embed_size = embed_size # d_model 
    self.heads = heads
    self.head_dim = embed_size // heads

    self.queries = nn.Linear(embed_size, embed_size ,bias=False)
    self.keys = nn.Linear(embed_size, embed_size, bias = False)
    self.values = nn.Linear(embed_size, embed_size, bias = False)

    self.dropout = nn.Dropout(dropout)
    self.fc_out = nn.Linear(embed_size, embed_size)

  def forward(self, values, keys, queries, mask=None):
    N = queries.shape[0]

    keys = self.keys(keys).view(N, -1, self.heads, self.head_dim)
    queries = self.queries(queries).view(N , -1, self.heads, self.head_dim)
    values = self.values(values).view(N, -1, self.heads, self.head_dim)

    keys = keys.transpose(1,2)
    queries = queries.transpose(1,2)
    values = values.transpose(1,2)


    # dot_product attention calculate
    energy = self.dot_product_attention(values, keys, queries, self.head_dim, mask, self.dropout)

    concat = energy.transpose(1,2).contiguous().view(N, -1, self.embed_size)
    out = self.fc_out(concat)

    return out

  def dot_product_attention(self, values, keys, queries, head_dim, mask=None, dropout=None):
    energy = torch.matmul(queries, keys.transpose(-2,-1) / math.sqrt(self.head_dim))

    if mask is not None:
      energy = energy.masked_fill(mask==0, -1e20)

    energy = torch.nn.functional.softmax(energy, dim=-1)

    if dropout is not None:
      energy = dropout(energy)
    out = torch.matmul(energy, values)
    return out

```
