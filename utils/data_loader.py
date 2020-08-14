from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from . import constant as model_config


device = model_config.device


class NERDataset(Dataset): 
    def __init__(self,dtype):
        self.text_seq = torch.load('./data/processed_data/{}_token_idx.pt'.format(dtype))
        self.entity_seq = torch.load('./data/processed_data/{}_ner_idx.pt'.format(dtype))
        
        assert len(self.text_seq)==len(self.entity_seq)
        self.length = len(self.text_seq)
        
    def __getitem__(self, idx):
        return self.text_seq[idx], self.entity_seq[idx]
    
    def __len__(self):
        return self.length
    
    
    
def pad_collate(batch):
    (xx, yy) = zip(*batch)

    original_token_len = torch.tensor([len(x) for x in xx]) # valid length
    original_label_len = torch.tensor([len(y) for y in yy])
    
    # print('maxlen is {}'.format(max(original_token_len)))
    token_ids = pad_sequences(xx, 
                              maxlen=model_config.maxlen, # max(original_token_len), 
                              value=model_config.pad_idx, 
                              padding='post',
                              dtype='long',
                              truncating='post')
    
    segment_ids = [len(i)*[0] for i in token_ids]

    label = pad_sequences(yy, 
                          maxlen=model_config.maxlen, # max(original_label_len), 
                          value=model_config.pad_label, 
                          padding='post', 
                          dtype='long',
                          truncating='post')
    
    return token_ids, original_token_len, segment_ids, label



def transform_to_bert_input(tokenized_idx_with_cls_sep, _device=None):
       
        if _device==None:
            _device = model_config.device
            
        
        token_ids = pad_sequences(tokenized_idx_with_cls_sep, 
                                  maxlen=model_config.maxlen,
                                  value=model_config.pad_idx, 
                                  padding='post',
                                  dtype='long',
                                  truncating='post')

        valid_length = torch.tensor([len(tokenized_idx_with_cls_sep[0])]) # .long()
        segment_ids = [len(tokenized_idx_with_cls_sep[0])*[0]]

        # torch-compatible format
        token_ids = torch.tensor(tokenized_idx_with_cls_sep).float().to(_device)
        valid_length = valid_length.clone().detach().to(_device)
        segment_ids = torch.tensor(segment_ids).long().to(_device)
        
        return token_ids, valid_length, segment_ids

