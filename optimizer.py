import numpy as np


class ScheduledOptim:
    
    def __init__(self, optimizer, warmup_steps):
        self.init_lr = np.power(params['embed_size'], -0.5)
        self.optimizer = optimizer
        self.step_num = 0
        self.warmup_steps = warmup_steps
    
    def step(self):
        self.step_num += 1 # per each step forward 
        lr = self.init_lr * self.get_scale() # scailing lr 
        
        for p in self.optimizer.param_groups: # Add a param group to the Optimizer s param_groups
            p['lr'] = lr
            
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_scale(self): # scaling lr 
        return np.min([
            np.power(self.step_num, -0.5), # step_num ** -0.5  
            self.step_num * np.power(self.warmup_steps, -1.5)
        ])
