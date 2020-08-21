from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, random, os, logging, glob
import torch
from torch.utils.data import DataLoader
from pytorch_transformers import AdamW, WarmupLinearSchedule
from kobert.pytorch_kobert import get_pytorch_kobert_model
from tqdm import tqdm, trange, tqdm_notebook, tnrange

from models.bert import KobertLSTMCRF, KobertCRF
from transformers import DistilBertModel
import utils.constant as model_config
from utils.log import logger, init_logger
from utils.data_loader import NERDataset, pad_collate
device = model_config.device

            
def transform_to_bert_input(batch):

    token_ids, valid_length, token_type_ids, label = batch[0], batch[1], batch[2], batch[3]

    token_ids = torch.from_numpy(token_ids).float().to(device) 
    valid_length = valid_length.clone().detach().to(device)
    token_type_ids = torch.tensor(token_type_ids).long().to(device)
    label =  torch.tensor(label).to(device)

    return token_ids, valid_length, token_type_ids, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_type", default='bert-crf', type=str, choices=['bert-crf', 'bert-lstm-crf'])
    parser.add_argument('-log', default='../logs/distill_3_kobert_crf_extend_whitespace.log')
    parser.add_argument('-log_dir', default='./result')
    parser.add_argument("-distill_layer", default=3, type=int, choices=[1,3,12])
    # parser.add_argument("-is_distill", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-epoch", default=model_config.epoch, type=int)
    parser.add_argument("-batch_size", default=model_config.batch_size, type=int)
    
    args = parser.parse_args()

    
    # LSTM
    if args.model_type == 'bert-lstm-crf':
        use_lstm='True'
    else:
        use_lstm='False'
    
    # Bert Model and Vocab
    kobert, vocab = get_pytorch_kobert_model()
    if args.distill_layer==1:
        kobert =  DistilBertModel.from_pretrained('./model/1_layer')
        is_distill = True
        args.log = '/layer_1_kobert_lstm_{}_crf_batch_{}_epoch_{}'.format(use_lstm, args.batch_size, args.epoch)
        
    elif args.distill_layer==3:
        kobert = DistilBertModel.from_pretrained('monologg/distilkobert')
        is_distill = True
        args.log = '/layer_3_kobert_lstm_{}_crf_batch_{}_epoch_{}'.format(use_lstm, args.batch_size, args.epoch)
    else:
        kobert = kobert
        is_distill = False
        args.log = '/layer_12_kobert_lstm_{}_crf_batch_{}_epoch_{}'.format(use_lstm, args.batch_size, args.epoch)
        
    # Make a log file
    log_path = args.log_dir+args.log
    init_logger(log_path, '/log/log.txt')

    # Load Entity Dictionary, Train and Test data
    entitiy_to_index = torch.load('./data/processed_data/entity_to_index.pt')
    index_to_entity = torch.load('./data/processed_data/index_to_entity.pt')

    print("Load processed data...")
    # Load process train and validation data
    train_dataset = NERDataset('train')
    valid_dataset = NERDataset('valid')

    # Build train_and validation loaders which generate data with batch_size
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=model_config.batch_size,
                              shuffle=True,
                              collate_fn=pad_collate,
                              drop_last=True,
                              num_workers=0)
    train_examples_len = len(train_loader)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=model_config.batch_size,
                              shuffle=True,
                              collate_fn=pad_collate,
                              drop_last = True,
                              num_workers=0)
    valid_examples_len = len(valid_loader)
    
    print("Build Model...")
    if args.model_type== 'bert-lstm-crf':
        model = KobertLSTMCRF(config = model_config, bert_model = kobert, distill= is_distill)
        model_name = 'bert_lstm_crf'
        logger.info(model_name)
    else:
        model = KobertCRF(config = model_config, bert_model = kobert, distill= is_distill)
        model_name = 'bert_crf'
        logger.info(model_name)
    
#     # Load Trained parameters
#     model_dict = model.state_dict()
#     model_save_path =model_save_path = '../models/bert_lstm_crf_normalized_extend_3/'
#     model_files = glob.glob(model_save_path+'*.pt')
#     best_acc_model = sorted(model_files, key=lambda x: x[-6:-3], reverse=True)[0]
#     print('Loading checkpoint from {}'.format(best_acc_model))
#     print(' ')

#     checkpoint = torch.load(best_acc_model)
#     model.load_state_dict(checkpoint['model_state_dict'])

    
    # Prepare optimizer and schedule (linear warmup and decay)
    print("Set Optimized and Scheduler...")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]
    
    # t_total = train_examples_len // model_config.gradient_accumulation_steps * model_config.epochs
    t_total = train_examples_len * args.epoch
    optimizer = AdamW(optimizer_grouped_parameters, lr=model_config.learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, model_config.warmup_steps, t_total)
     
    # Train
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", train_examples_len)
    logger.info("  Num validation examples = %d", valid_examples_len)
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Total steps = %d", t_total)
    
    global_step = 0
    best_eval_acc = 0.0
    
#     best_steps = 0
#     best_tr_acc, best_tr_loss = 0.0, 99999999999.0
#     tr_loss, logging_loss = 0.0, 0.0
#     best_dev_acc, best_dev_loss = 0.0, 99999999999.0

        
    for epoch in tqdm(range(args.epoch)):
        
        """ Train Step """
        model.train()
        model.to(device)
        tr_acc, tr_loss = 0.0, 0.0
        
        for step, batch in enumerate(train_loader):
            global_step += 1
            (token_ids, valid_length, segment_ids, label) = transform_to_bert_input(batch)

            # model output
            log_likelihood, sequence_of_tags = model(token_ids, valid_length, segment_ids, label)
            loss = -1 * log_likelihood

#             if n_gpu > 1:
#                 loss = loss.mean() # mean() to average on multi-gpu parallel training
#             if model_config.gradient_accumulation_steps > 1:
#                 loss = loss / model_config.gradient_accumulation_steps

            tr_loss += loss.item()  
            loss.backward()

            with torch.no_grad():
                sequence_of_tags = torch.tensor(sequence_of_tags).to(device)
                accuracy = (sequence_of_tags==label).float()[label != model_config.pad_label].mean()
                tr_acc += accuracy.item()
                
            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)

            if (step) % model_config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step() # update learning rate
                model.zero_grad()
                optimizer.zero_grad()
                global_step += model_config.gradient_accumulation_steps
                
            
            if global_step % 100 == 0:
                logger.info('epoch : {}, global_step : {} / {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch+1, global_step, t_total, loss.item() , accuracy.item()))
            
        tr_avg_acc = tr_acc / (step+1)
        tr_avg_loss = tr_loss / (step+1)
        logger.info('[Train]      epoch : {}, global_step : {} / {}, tr_avg_loss: {:.3f}, tr_avg_acc: {:.2%}'.format(epoch+1, global_step, t_total, tr_avg_loss, tr_avg_acc))
        
        
        """ Evaluation Step"""
        model.eval()
        eval_acc, eval_loss = 0.0, 0.0
       
        for step, batch in enumerate(valid_loader):
            (token_ids, valid_length, segment_ids, label) = transform_to_bert_input(batch)

            # model output
            log_likelihood, sequence_of_tags = model(token_ids, valid_length, segment_ids, label)
            loss = -1 * log_likelihood
            eval_loss += loss.item()  

            with torch.no_grad():
                sequence_of_tags = torch.tensor(sequence_of_tags).to(device)
                accuracy = (sequence_of_tags==label).float()[label != model_config.pad_label].mean()
                eval_acc += accuracy.item()

        eval_avg_acc = eval_acc / (step+1)
        eval_avg_loss = eval_loss / (step+1)
        logger.info('[Evaluation] epoch : {}, Eval_Loss: {:.3f}, Eval_Acc: {:.2%}'.format(epoch+1, eval_avg_loss, eval_avg_acc))
        
       
        """ Model save"""
        if eval_avg_acc > best_eval_acc:
            model.to('cpu')
            best_eval_acc = eval_avg_acc
            state = {'epoch':epoch+1,'model_state_dict': model.state_dict()}
            save_path = '{}/epoch_{}_step_{}_tr_acc_{:.3f}_eval_acc_{:.3f}.pt'.format(
                        log_path, epoch+1, global_step, tr_avg_acc, eval_avg_acc)
            
            if len(glob.glob(log_path+'/epoch*.pt'))>0:
                os.remove(glob.glob(log_path+'/epoch*.pt')[0])

            torch.save(state, save_path)
            logger.info('Model saving with best eval acc : {:.2%}'.format(eval_avg_acc))
            
            # if len(glob.glob(log_path+'/epoch*.pt'))==0:
            os.mkdir(log_path+'/epoch_{}_step_{}_tr_acc_{:.3f}_eval_acc_{:.3f}'.format(epoch+1, global_step, tr_avg_acc, eval_avg_acc))