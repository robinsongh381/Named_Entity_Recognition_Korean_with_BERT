from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import utils.constant as model_config


def evaluate(model, valid):
    """evaluate accuracy and loss against validation data"""
    
    model.eval()
    device = model_config.device
    results = {}

    eval_loss = 0.0
    nb_eval_steps = 0
    
    list_of_true_label = []
    list_of_pred_label = []
    count_correct = 0
    
    print('*****************Evaluation*****************')
    # epoch_iterator = tqdm(valid, desc="Evaluation")
    # for step, batch in enumerate(epoch_iterator):
    for step, batch in enumerate(valid):
        (token_ids, valid_length, segment_ids, label) = batch
        
        # no grad
        with torch.no_grad():

            # move gpu for values to be updated
            token_ids = torch.from_numpy(token_ids).float().to(device)
            valid_length = valid_length.clone().detach().to(device)
            segment_ids = torch.tensor(segment_ids).long().to(device)
            label =  torch.tensor(label).to(device)

            log_likelihood, predict_label = model(token_ids,
                                                  valid_length,
                                                  segment_ids,
                                                  label)
            eval_loss += -1 * log_likelihood

        nb_eval_steps+=1

        # true_label = torch.tensor(label).to('cpu')
        true_label = label.clone().detach().to('cpu')
        predict_label = torch.tensor(predict_label).to('cpu')
        # predict_label = predict_label.clone().detach().to('cpu')
        
        count_correct += (predict_label==true_label).float()[true_label != model_config.pad_label].mean()

        for l in true_label.tolist():
            list_of_true_label+= l

        for l in predict_label.tolist():
            list_of_pred_label+= l 


    eval_loss = eval_loss / nb_eval_steps
    acc = count_correct / nb_eval_steps
    
    result = {"eval_loss": "{:.3f}".format(eval_loss.item()), 
              "eval_acc": "{:.2%}".format(acc.item())}
    results.update(result)
    
    return results, list_of_true_label, list_of_pred_label