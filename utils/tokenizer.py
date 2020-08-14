from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences



class Tokenizer:
    """ Tokenizer class"""

    def __init__(self, split_fn):
        # self._vocab = vocab
        self.tokenizer = split_fn
        self.cls_idx = self.tokenizer.vocab.to_indices('[CLS]')
        self.sep_idx = self.tokenizer.vocab.to_indices('[SEP]')
 
        
    def __call__(self, text_string):
        return self.tokenizer(text_string)
    
    
    def sentencepiece_tokenizer(self, raw_text):
        return self.tokenizer(raw_text)
    
    
    def token_to_cls_sep_idx(self, text_list):
        
        # tokenized_text_list = sentencepiece_tokenizer(text)
        idx_tok = []
        for t in text_list:
            idx = self.tokenizer.convert_tokens_to_ids(t)
            idx_tok.append(idx)
        idx_tok = [self.cls_idx] + idx_tok + [self.sep_idx]

        return idx_tok