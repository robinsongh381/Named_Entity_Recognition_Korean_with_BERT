import glob, re
import torch
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp

from utils.tokenizer import Tokenizer


# Bert Model and Vocab
_, vocab = get_pytorch_kobert_model()

# Tokenizer
_tok_path = get_tokenizer()
_pretrained_tokenizer = nlp.data.BERTSPTokenizer(_tok_path, vocab, lower=False)
tokenizer = Tokenizer(_pretrained_tokenizer)

# Entitiy-index dictionary
global_entity_dict = torch.load('./data/processed_data/entity_to_index.pt')

# Load raw data 
train_set = glob.glob('./data/raw_data/train_set/*.txt')
valid_set = glob.glob('./data/raw_data/validation_set/*.txt')


def ner_tag_to_idx(ner_list):
    idx_tag = []
    for tag in ner_list:
        idx = global_entity_dict[tag]
        idx_tag.append(idx)
    # add tag for [CLS] and [SEP]
    idx_tag = [global_entity_dict['O']]+idx_tag+[global_entity_dict['O']]
    
    return idx_tag


def transform_source_fn(raw_text):
    prefix_sum_of_token_start_index = []
    
    # tokens = tok(text)
    tokenized_text = tokenizer(raw_text)
    sum = 0
    for i, token in enumerate(tokenized_text):
        if i == 0:
            prefix_sum_of_token_start_index.append(0)
            sum += len(token) - 1
        else:
            prefix_sum_of_token_start_index.append(sum)
            sum += len(token)
    return tokenized_text, prefix_sum_of_token_start_index


def transform_target_fn(label_text, tokens, prefix_sum_of_token_start_index):

    regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
    regex_filter_res = regex_ner.finditer(label_text)

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []


    count_of_match = 0
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]  # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match  # delete previous '<, :, 3 words tag name, >'
        end_index = match_item.end() - 6 - 6 * count_of_match

        list_of_ner_tag.append(ner_tag)
        list_of_ner_text.append(ner_text)
        list_of_tuple_ner_start_end.append((start_index, end_index))
        count_of_match += 1

    list_of_ner_label = []
    entity_index = 0
    is_entity_still_B = True
    for tup in zip(tokens, prefix_sum_of_token_start_index):
        token, index = tup

        if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
            index += 1    # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

        if entity_index < len(list_of_tuple_ner_start_end):
            start, end = list_of_tuple_ner_start_end[entity_index]

            if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                is_entity_still_B = True
                entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                start, end = list_of_tuple_ner_start_end[entity_index]

            if start <= index and index < end:  
                entity_tag = list_of_ner_tag[entity_index]
                if is_entity_still_B is True:
                    entity_tag = 'B-' + entity_tag
                    list_of_ner_label.append(entity_tag)
                    is_entity_still_B = False
                else:
                    entity_tag = 'I-' + entity_tag
                    list_of_ner_label.append(entity_tag)
            else:
                is_entity_still_B = True
                entity_tag = 'O'
                list_of_ner_label.append(entity_tag)
        else:
            entity_tag = 'O'
            list_of_ner_label.append(entity_tag)
    
    return list_of_ner_label


# Process raw data and save indexed .pt files
reg_label = re.compile('<(.+?):[A-Z]{3}>') # detect texts with entity tag
reg_idx = re.compile('## \d+$') # detect texts without entity tag

train_set_token_idx_list = []
train_set_ner_idx_list = []
valid_set_token_idx_list = []
valid_set_ner_idx_list = []

mode = ['train', 'valid']
for m in mode:
    if m=='train':
        dataset = train_set
        save_token_list = train_set_token_idx_list
        save_ner_list = train_set_ner_idx_list
        print('Processing {} training data...'.format(len(dataset)))
    else:
        dataset = valid_set
        save_token_list = valid_set_token_idx_list
        save_ner_list = valid_set_ner_idx_list
        print('Processing {} validation data...'.format(len(dataset)))
    
    token_count = 0
    ner_count = 0
    
    for file in dataset:
        with open(file, "r", encoding = "utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n','')

                if reg_idx.search(line): ## 1
                    continue

                elif line[:2]=='##' and not reg_label.search(line) : # raw_text
                    token_count+=1                   
                    
                    raw_text = line[3:]
                    tokenized_text, start_index = transform_source_fn(raw_text)
                    cls_sep_idx = tokenizer.token_to_cls_sep_idx(tokenized_text)
                    save_token_list.append(cls_sep_idx)

                elif line[:2]=='##' and reg_label.search(line): # label_text
                    ner_count+=1
                    assert token_count==ner_count

                    label_text = line[3:]
                    ner_tag = transform_target_fn(label_text, tokenized_text, start_index)
                    cls_sep_ner_idx_tag = ner_tag_to_idx(ner_tag)

                    assert len(cls_sep_idx)==len(cls_sep_ner_idx_tag)
                    save_ner_list.append(cls_sep_ner_idx_tag)
                    
    
    # Save processed data to .pt files
    torch.save(save_token_list, './data/processed_data/{}_token_idx.pt'.format(m))
    torch.save(save_ner_list, './data/processed_data/{}_ner_idx.pt'.format(m))
    print('{} files saved to ./data/processed_data/{}_token_idx.pt'.format(m, m))