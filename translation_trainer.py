# trainer class 

import torch
import random

from model.transformer import Transformer
from optimizer import ScheduledOptim
from params import 

from tqdm.auto import tqdm  


random.seed(0)
torch.manual_seed(0)




class TransTrainer:
    def __init__(self, mode, params, train_iter=None, valid_iter=None, test_iter=None):
        # mode: (str) 'train' / 'test'
        # params: (dict) 
        # train/valid/test_iter: (iterator)
        
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Transformer(self.params['src_vocab_size'],
                                 self.params['trg_vocab_size'],
                                 self.params['src_pad_idx'],
                                 self.params['trg_pad_idx'],
                                 self.params['embed_size'],
                                 self.params['num_layers'],
                                 self.params['forward_expansion'],
                                 self.params['heads'],
                                 self.params['dropout'],
                                 self.params['max_length'],
                                 self.params['device']).to(self.device)
        
        print(self.model)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index = self.params['trg_pad_idx'])
        

        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter
        else: # 'test'
            self.test_iter = test_iter
        
        print('initializing weights..')
        self.model.apply(self.initialize_weights)
        # print(self.model)
        print(f'The model has {self.count_parameters(self.model):,} trainable parameters')

    # num_param count
    def count_parameters(self,model):  
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    # weight initialization
    def initialize_weights(self,m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)



    def train(self, epochs = 10, optimizer = 'Adam', clip=1):
        
        if optimizer == 'Adam':
            optimizer = ScheduledOptim(torch.optim.Adam(self.model.parameters(),
                                                        betas = [0.9, 0.98],
                                                        eps = 1e-9),
                                                        warmup_steps=4000
                                                        )
        
        elif optimizer == 'SGD':
            optimizer = ScheduledOptim(torch.optim.SGD(self.model.parameters(),lr=5e-4))
                                                      
                                                      
                                                        

        best_valid_loss = float('inf')

        print('start training..')
        for epoch in tqdm(range(epochs)):
            self.model.train() # train_mode 
            epoch_loss = 0

            for batch in tqdm(self.train_iter): 
                src = batch.src
                trg = batch.trg

                optimizer.zero_grad()
                output = self.model(src, trg[:, :-1]) # trg except <EOS> token
                
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                # output = [(batch * trg_length -1) , output_dim]
                # trg = [(batch * trg_length -1)]

                loss = self.criterion(output, trg)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                
                optimizer.step()
                epoch_loss += loss.item()
        
            train_loss = epoch_loss / len(self.train_iter)
            valid_loss = self.evaluate()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), "transformer_ko_to_en.pt")

            print(f'Epoch: {epoch + 1:02}')
            print(f'\tTrain Loss: {train_loss: .3f}')
            print(f'\tValid Loss: {valid_loss: .3f}')
    
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_iter):
                src = batch.src
                trg = batch.trg
 
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = self.criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_iter)


    def test(self):
        self.model.load_state_dict(torch.load('transformer_ko_to_en.pt'))
        self.model.eval()
        epoch_loss=0

        with torch.no_grad():
            for batch in self.test_iter:
                src = batch.src
                trg = batch.trg
 
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = self.criterion(output, trg)

                epoch_loss += loss.item()

        test_loss = epoch_loss / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f}')
