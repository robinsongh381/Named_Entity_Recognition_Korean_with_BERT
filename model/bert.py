import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .crf import CRF
# from torchcrf import CRF


class KobertLSTMCRF(nn.Module):
    def __init__(self, config, bert_model, distill=False):
        super(KobertLSTMCRF, self).__init__()
        
        self.distill = distill
        self.config = config
        self.bert = bert_model
        self.dropout = nn.Dropout(self.config.dropout)
        self.bilstm  = nn.LSTM(self.config.hidden_size, self.config.hidden_size //2, 
                               batch_first=True, bidirectional=True )
        self.linear = nn.Linear(self.config.hidden_size, self.config.num_class)
        self.crf = CRF(num_tags=self.config.num_class, batch_first=True)
        
    def get_attention_mask(self, input_ids, valid_length):
        attention_mask = torch.zeros_like(input_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, input_ids, valid_length=None, token_type_ids=None, tags=None):
        attention_mask = self.get_attention_mask(input_ids, valid_length)
        
        if self.distill:
            outputs = self.bert(input_ids=input_ids.long(),
                                                      attention_mask=attention_mask)
            all_encoder_layers = outputs[0]

            
        else:
            all_encoder_layers, pooled_output = self.bert(input_ids=input_ids.long(),
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        
        last_encoder_layer = all_encoder_layers
        drop = self.dropout(last_encoder_layer)
        
        if self.config.pack_sequence:
            # @ TODO: set config.maxlen ~ 30
            packed_drop = pack_padded_sequence(drop, valid_length.cpu().numpy(), batch_first=True, enforce_sorted=False)
            packed_output, hc = self.bilstm(packed_drop)
            unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=self.config.pad_idx)
            linear = self.linear(unpacked_output)
         
        else:
            output, hc = self.bilstm(drop)
            linear = self.linear(output)

        if tags is not None: # crf training
            log_likelihood = self.crf(linear, tags)
            tag_seq = self.crf.decode(linear)
            
            return log_likelihood, tag_seq
        
        else: # for inference
            tag_seq = self.crf.decode(linear)
            confidence = self.crf.compute_confidence(linear, tag_seq)

            return tag_seq, confidence
        
        
class KobertCRF(nn.Module):
    def __init__(self, config, bert_model, distill=False):
        super(KobertCRF, self).__init__()
        
        self.distill = distill
        self.bert = bert_model
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.num_class)
        self.crf = CRF(num_tags=config.num_class, batch_first=True)
    
    def get_attention_mask(self, input_ids, valid_length):
        attention_mask = torch.zeros_like(input_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, input_ids, valid_length, token_type_ids, tags=None):
        attention_mask = self.get_attention_mask(input_ids, valid_length)

        if self.distill:
            all_encoder_layers, pooled_output = self.bert(input_ids=input_ids.long(),
                                                      attention_mask=attention_mask)
        else:
            all_encoder_layers, pooled_output = self.bert(input_ids=input_ids.long(),
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        
        
        # print('all_encoder_layers')
        # print(all_encoder_layers.size()) # [batch, maxlen, hidden_size]
        # print('***********')
        
        last_encoder_layer = all_encoder_layers
        drop = self.dropout(last_encoder_layer)
        linear = self.linear(drop)

        if tags is not None: # crf training
            log_likelihood = self.crf(linear, tags)
            tag_seq = self.crf.decode(linear)
            
            return log_likelihood, tag_seq
        
        else: # for inference
            tag_seq = self.crf.decode(linear)
            confidence = self.crf.compute_confidence(linear, tag_seq)

            return tag_seq, confidence
        
        
class Kobert(nn.Module):
    def __init__(self, config, bert_model):
        super(Kobert, self).__init__()
        
        self.bert = bert_model
        self.dropout = nn.Dropout(config.dropout)
        # self.linear = nn.Linear(config.hidden_size, config.num_class)
        
    def get_attention_mask(self, input_ids, valid_length):
        attention_mask = torch.zeros_like(input_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, input_ids, valid_length, token_type_ids, tags=None):
        attention_mask = self.get_attention_mask(input_ids, valid_length)

        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids.long(),
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask)
        
        return pooled_output
        