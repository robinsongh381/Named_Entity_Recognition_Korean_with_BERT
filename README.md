# Named Entity Recognition
**Features**
  - Use the pretrained Korean BERT (KoBERT) from [SKT Brain](https://github.com/SKTBrain/KoBERT)
  - The tokenizer used is also from KoBERT with vocab size 8002
  - The data is provided from [한국해양대학교 자연언어처리 연구실](https://github.com/kmounlp/NER), with 23032 training sentences and 931 for validation
  - BIO-tagging scheme is applied with 10 distinct entity labels (hence total 21 possible labels)
  - Experimented with two models, one with KoBERT-FF-CRF and the other KoBERT-FF-LSTM-CRF
  - The result for iterating 12 epochs
  
|        | Training Acc | Validation Acc | 
| ------ | ------ |------ |
| with LSTM | **97.2%** | 93.3% |
| without LSTM  | **98.0%** | 94.1% |


**Python version**: This code is in Python3.6 
 
**Updates**: An option for removing 조사 within an entity text (eg. 홍길동은(PER)->홍길동(PER)) is provided based on Komoran pos tagger


# Steps for Training Model
Please note that all configurations for training model (hyper_parameters, save_path, load_path, etc) is defined at [constant.py](./utils/constant.py)


## Data Process

```
python preprocess.py 
```
By default, all raw data is located at `./data/raw_data` and the processed data after running `preprocess.py` will be saved at `./data/processed_data` in the form of `.pt` files

## Model Training
```
python train.py -model_type bert-lstm-crf -log ./logs/bert-lstm-crf.log
```
The `model_type` option should be either `bert-crf` or `bert-lstm-crf` depending your choice for the model.
Logs for training and evaluation will be recorded at `./logs/bert-lstm-crf.log`

Please note that you can change batch_size, epochs etc from [constant.py](./utils/constant.py)

## Inference
```
python inference_cmd.py -model bert-lstm-crf
```

This will load a saved model with the highest accuracy from `../models/bert-lstm-crf`.
