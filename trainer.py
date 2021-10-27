# data imports
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# model imports
# similar to https://github.com/huggingface/transformers/blob/14e9d2954c3a7256a49a3e581ae25364c76f521e/src/transformers/models/bert/modeling_bert.py
import logging

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, HingeEmbeddingLoss

#from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertPreTrainedModel
from transformers.utils import logging

# from transformers.file_utils import ModelOutput
from typing import Optional

# logger = logging.get_logger(__name__)

# Trainer imports
from transformers import Trainer, TrainingArguments
from dataset import ArgumentDataset
from model import BertForPhraseClassification
from sklearn.metrics import classification_report

from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # print(logits.shape, labels.shape)
    predictions = np.argmax(logits, axis=-1)
    print(predictions[0])
    labels = [l for para_l in labels for l in para_l if l != -100]
    predictions = [p for para_p in predictions for p in para_p if p != -100]
    print(len(labels), len(predictions))
    print(classification_report(labels, predictions))
    return metric.compute(predictions=predictions, references=labels)

def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    edus_files, labels_files = '../data/training_data/*.edus', '../data/training_data/*.labels'
    argdata = ArgumentDataset(tokenizer, edus_files, labels_files)

    # edu_sep_id = tokenizer.convert_tokens_to_ids('[EDU_SEP]')
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=5)
    edu_tag_model = BertForPhraseClassification.from_pretrained("bert-base-uncased", config=config)
    edu_tag_model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir='./',      
        num_train_epochs=1,
        per_device_train_batch_size=32,  
        save_steps=0, 
        do_train=True,
        do_eval=True,
        dataloader_drop_last=True,
        logging_steps=50
    )

    trainer = Trainer(
        model=edu_tag_model,        
        args=training_args,                
        train_dataset=argdata,
        eval_dataset=argdata,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=argdata)
    print(metrics)

if __name__ == '__main__':
    main()
