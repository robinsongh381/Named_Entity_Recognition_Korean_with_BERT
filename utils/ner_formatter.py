
import re
from konlpy.tag import Komoran  # Komoran is relatively faster to load
from . import constant as model_config

komoran = Komoran() 


def remove_josa(text):
    
    list_of_pos = komoran.pos(text)
    if list_of_pos[-1][1][0]=='J':
        text=text.replace(list_of_pos[-1][0],'')
    
    return text
        

def compute_found_ner(sentence_with_tag, confidence):
    
    count_of_match = 0
    confidence = round(confidence,3)
    list_of_ner_word=[]    
    
    regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 
    regex_filter_res = regex_ner.finditer(sentence_with_tag)
    
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]        # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match
        if model_config.remove_josa:
            ner_text = remove_josa(ner_text) 
        end_index = start_index + len(ner_text)
        # end_index = match_item.end() - 6 - 6 * count_of_match 

        list_of_ner_word.append({"start": start_index,
                             "end": end_index,
                             "value": ner_text, # entity_word.replace("▁", " "), 
                             "entity": ner_tag, 
                             "confidence": confidence,
                             "extractor": "global_entity_extractor"}
                           )
        count_of_match+=1
        
    return list_of_ner_word


def simple_compute_found_ner(sentence_with_tag, confidence):
    
    count_of_match = 0
    confidence = round(confidence,3)
    list_of_ner_word=[]    
    
    regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 
    regex_filter_res = regex_ner.finditer(sentence_with_tag)
    
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]        # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match
        if model_config.remove_josa:
            ner_text = remove_josa(ner_text) 
        end_index = start_index + len(ner_text)
        # end_index = match_item.end() - 6 - 6 * count_of_match 

        list_of_ner_word.append({
                             "value": ner_text,
                             "entity": ner_tag
                                })
        count_of_match+=1
        
    return list_of_ner_word


def decoding_text_with_tag(input_token, pred_ner_tag):
    
    input_token = ['[CLS]']+input_token+['[SEP]']
    
    decoding_ner_sentence = ""
    is_prev_entity = False
    prev_entity_tag = ""
    is_there_B_before_I = False

    for token_str, pred_ner_tag_str in zip(input_token, pred_ner_tag):
        token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체
        if 'B-' in pred_ner_tag_str:
            if is_prev_entity is True:
                decoding_ner_sentence += ':' + prev_entity_tag+ '>'

            if token_str[0] == ' ':
                token_str = list(token_str)
                token_str[0] = ' <'
                token_str = ''.join(token_str)
                decoding_ner_sentence += token_str
            else:
                decoding_ner_sentence += '<' + token_str
            is_prev_entity = True
            prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
            is_there_B_before_I = True

        elif 'I-' in pred_ner_tag_str:
            decoding_ner_sentence += token_str

            if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                is_prev_entity = True
        else:
            if is_prev_entity is True:
                decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                is_prev_entity = False
                is_there_B_before_I = False
            else:
                decoding_ner_sentence += token_str
                
    decoding_ner_sentence=decoding_ner_sentence.replace('[CLS]','').replace('[SEP]','').rstrip().lstrip()
    
    # 120 , 000 -> 120,000
    decoding_ner_sentence = decoding_ner_sentence.replace(' ,',',').replace(', ','')
                
    return decoding_ner_sentence