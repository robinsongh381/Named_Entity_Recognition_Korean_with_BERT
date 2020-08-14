from __future__ import absolute_import
import torch

# entitiy_to_index = torch.load('../data/processed_data/entity_to_index.pt')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hidden_size = 768
maxlen = 128
epochs = 5
batch_size = 32
dropout = 0.1
learning_rate = 5e-5
warmup_proportion = 0.1
gradient_accumulation_steps = 5
summary_step = 250
adam_epsilon = 1e-8
warmup_steps = 0
max_grad_norm = 1

remove_josa = True
remove_special_char = True
model_dir = "../models"

eval_during_train = True
display_eval_plots = False
evaluate_step = 1000
save_steps = 1000

pack_sequence = False
num_class = 22   # len(entitiy_to_index)
pad_idx = 1      # tok.convert_tokens_to_ids('[PAD]')
pad_label = 21   # entitiy_to_index['[PAD]']
o_label = 20     # entitiy_to_index['O']
