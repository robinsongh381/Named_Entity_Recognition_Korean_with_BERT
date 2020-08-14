from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, random, os, logging, glob
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from pytorch_transformers import AdamW, WarmupLinearSchedule
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp as nlp
from tqdm import tqdm, trange, tqdm_notebook, tnrange

from models.bert import KobertLSTMCRF, KobertCRF
from transformers import DistilBertModel
from evaluate import evaluate
import utils.constant as model_config
from utils.tokenizer import Tokenizer
from utils.data_loader import NERDataset, pad_collate
from utils.manager import CheckpointManager
from utils.log import logger, init_logger
device = model_config.device


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default='bert-crf', type=str, choices=['bert-crf', 'bert-lstm-crf'])
    parser.add_argument('-log', default='../logs/distill_3_kobert_crf_extend_whitespace.log')
    parser.add_argument("-distill_layer", default=12, type=int, choices=[1,3,12])
    # parser.add_argument("-is_distill", type=str2bool, nargs='?',const=True,default=False)
    
    args = parser.parse_args()

    
    # Bert Model and Vocab
    kobert, vocab = get_pytorch_kobert_model()
    if args.distill_layer==1:
        kobert =  DistilBertModel.from_pretrained('./model/1_layer')
        is_distill = True
        args.log = './logs/distill_1_kobert_crf.log'
        
    elif args.distill_layer==3:
        kobert = DistilBertModel.from_pretrained('monologg/distilkobert')
        is_distill = True
        args.log = './logs/distill_3_kobert_crf.log'
    else:
        kobert = kobert
        is_distill = False
        args.log = './logs/distill_12_kobert_crf.log'
     
    # Make a log file
    init_logger(args.log)

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
    if args.model== 'bert-lstm-crf':
        model = KobertLSTMCRF(config = model_config, bert_model = kobert, distill= is_distill)
        model.to(device)
        model_name = 'bert_lstm_crf'
        logger.info(model_name)
    else:
        model = KobertCRF(config = model_config, bert_model = kobert, distill= is_distill)
        model.to(device)
        model_name = 'bert_crf'
        logger.info(model_name)
    
    # Load Trained parameters
    model_dict = model.state_dict()
    model_save_path =model_save_path = '../models/bert_lstm_crf_normalized_extend_3/'
    model_files = glob.glob(model_save_path+'*.pt')
    best_acc_model = sorted(model_files, key=lambda x: x[-6:-3], reverse=True)[0]
    print('Loading checkpoint from {}'.format(best_acc_model))
    print(' ')

    checkpoint = torch.load(best_acc_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    
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
    t_total = train_examples_len * model_config.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=model_config.learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, model_config.warmup_steps, t_total)
    
    # save
    model_config.model_dir = "../models"
    model_save_dir = model_config.model_dir+'/'+model_name+'_normalized_extend_whitespace_'+str(args.distill_layer)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    checkpoint_manager = CheckpointManager(model_save_dir)
     
    # Train
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", train_examples_len)
    logger.info("  Num validation examples = %d", valid_examples_len)
    logger.info("  Num Epochs = %d", model_config.epochs)
    logger.info("  Batch size = %d", model_config.batch_size)
    logger.info("  Total steps = %d", t_total)
    logger.info("  Model save directory = %s", model_save_dir)
    
    n_gpu = torch.cuda.device_count()
    global_step = 0
    best_steps = 0
    best_tr_acc, best_tr_loss = 0.0, 99999999999.0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0

#     for e in tnrange(model_config.epochs, desc="Epochs"):
#         epoch_iterator = tqdm(train_loader, desc="iteration")
        
    for e in tqdm(range(model_config.epochs)):
        epoch_iterator = tqdm(train_loader)
        epoch = e
        for step, batch in enumerate(epoch_iterator):
            model.train()

            (token_ids, valid_length, segment_ids, label) = batch

            # set all delta_grad to zero
            optimizer.zero_grad()

            # move gpu for values to be updated
            token_ids = torch.from_numpy(token_ids).float().to(device)
            valid_length = valid_length.clone().detach().to(device)
            segment_ids = torch.tensor(segment_ids).long().to(device)
            label =  torch.tensor(label).to(device)

            # model output
            log_likelihood, sequence_of_tags = model(token_ids,
                                                     valid_length,
                                                     segment_ids,
                                                     label)

            loss = -1 * log_likelihood

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if model_config.gradient_accumulation_steps > 1:
                loss = loss / model_config.gradient_accumulation_steps

            tr_loss += loss.item()  
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)

            if (step) % model_config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step() # update learning rate
                model.zero_grad()
                global_step += model_config.gradient_accumulation_steps

                with torch.no_grad():
                    
                    sequence_of_tags = torch.tensor(sequence_of_tags).to(device)
                    accuracy = (sequence_of_tags==label).float()[label != model_config.pad_label].mean()

                tr_acc = accuracy.item()
                tr_loss_avg = tr_loss / global_step
                tr_summary = {'train_loss': tr_loss_avg, 'train_acc': tr_acc}

                logger.info('epoch : {}, global_step : {} /{},tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, global_step, t_total, tr_summary['train_loss'], tr_summary['train_acc']))

                # Evaluate
                if global_step>0 and global_step % model_config.evaluate_step ==0 and model_config.eval_during_train:
                    eval_summary, true_label, pred_label = evaluate(model,valid_loader)
                    logger.info("Evaluation at epoch : {}, global step: {}, {}".format(epoch+1, global_step, eval_summary ))
                    logger.info('')
                    logger.info(eval_summary)
                    logger.info('')
                    logger.info('*'*45)
                    # print classification report and save confusion matrix
                    if model_config.display_eval_plots:
                        logger.info('*********** Display Classification report and Confusion Matrix ***********')
                        cr_save_path = '{}/best-eval-epoch-{}-step-{}-acc-{}-cr.csv'.format(model_save_dir, epoch + 1, global_step, eval_summary['eval_acc'])
                        cm_save_path = '{}/best-eval-epoch-{}-step-{}-acc-{}-cm.png'.format(model_save_dir, epoch + 1, global_step, eval_summary['eval_acc'])
                        save_cr_and_cm(index_to_entity, 
                                       true_label, pred_label, 
                                       cr_save_path=cr_save_path, 
                                       cm_save_path=cm_save_path)

                # Model Save   
                if model_config.save_steps > 0 and global_step % model_config.save_steps == 0:

                    # Check best_tr_acc
                    is_tr_best =tr_acc >= best_tr_acc
                    if is_tr_best:
                        best_steps=global_step
                        best_tr_acc=tr_acc
                        logger.info("Saving model checkpoint as best-epoch-{}-step-{}-acc-{:.3f}.pt at {}".format(epoch + 1, global_step, tr_acc, model_save_dir))
                        
                        state = {'global_step':global_step+1,
                                 'model_state_dict':model.state_dict(),
                                 'opt_state_dict':optimizer.state_dict()}

                        ckpt_state = 'best-train-epoch-{}-step-{}-acc-{:.3f}.pt'.format(epoch + 1, global_step, tr_acc)
                        checkpoint_manager.save_checkpoint(state,ckpt_state)
                        logger.info("Saving best during train at epoch : {}, global step: {}, accuracy : {:.3f}".format(epoch+1, global_step, tr_acc))

    logger.info("[Final Result] epoch = {}, global_step = {}, average loss = {}, accuracy = {}".format(epoch+1, global_step, tr_loss / global_step, tr_acc))
    
    eval_summary, _, _ = evaluate(model,valid_loader)
    logger.info('[Final Evaluation]')
    logger.info(eval_summary)