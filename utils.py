import torch
from torchtext.data import TabularDataset, Field, BucketIterator

from tqdm.auto import tqdm


# make dataset
def convert_to_dataset(data_path, src_field, trg_field):
    dataset = TabularDataset(path = data_path, format = 'csv',
                             fields = [('src', src_field), ('trg', trg_field)])
    return dataset



# make data iterator
def make_iterator(batch_size,
                  mode,
                  src_field, trg_field,
                  train_path=None, valid_path=None, test_path=None):
    # Args
    # batch_size: (int)
    # mode: (str) 'train'/'test' 
    # train/valid/test_path: (str) csv_data_dir
  
    # file_src = open('src.pickle', 'rb')
    # src = pickle.load(file_src)
    # file_trg = open('trg.pickle', 'rb')
    # trg = pickle.load(file_trg)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  
    if mode == 'train': # use convert_to_dataset func
      train_dataset = convert_to_dataset(train_path, src_field, trg_field)
      valid_dataset = convert_to_dataset(valid_path, src_field, trg_field)

      # use BucketIterator to make iterator 
      train_iterator, valid_iterator = BucketIterator.splits(datasets = (train_dataset,valid_dataset),
                                                             batch_sizes= (batch_size,batch_size),
                                                             device=device,
                                                             sort=False)

      return train_iterator, valid_iterator


    else: # 'test'
      test_dataset = convert_to_dataset(test_path, src_field, trg_field)
      test_iterator = BucketIterator(test_dataset,
                                     batch_size = batch_size,
                                     device = device,
                                     sort=False)
      return test_iterator




