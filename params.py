import torch

class ModelParams:
    def __init__(self,
                 embed_size=256,
                 num_layers=3,
                 forward_expansion=2,
                 heads=8,
                 dropout=0.2,
                 max_length=100):
        
        # have default 
        self.embed_size = embed_size # hidden_dim
        self.num_layers = num_layers 
        self.forward_expansion = forward_expansion 
        self.heads = heads # num_heads
        self.dropout = dropout # dropout_ratio
        self.max_length = max_length
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
       
    def load_vocab_params(self,src_field, trg_field):
        
        src_vocab_size = len(src_field.vocab)
        trg_vocab_size = len(trg_field.vocab)
        src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
        trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]

        params = {'src_vocab_size': src_vocab_size,
                  'trg_vocab_size': trg_vocab_size,
                  'src_pad_idx': src_pad_idx,
                  'trg_pad_idx': trg_pad_idx,

                  'embed_size': self.embed_size,
                  'num_layers': self.num_layers,
                  'forward_expansion': self.forward_expansion,
                  'heads':self.heads,
                  'dropout': self.dropout,
                  'max_length': self.max_length,
                  'device': self.device}
        
        return params
