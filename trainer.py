import os
import argparse

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
    print(labels[0])
    labels = [l for para_l in labels for l in para_l]
    predictions = [p for para_p in predictions for p in para_p]
    print(len(labels), len(predictions))
    print(classification_report(labels, predictions))
    return metric.compute(predictions=predictions, references=labels)

def main():

    parser = argparse.ArgumentParser(description='Retrofitting on a single paraphrase dataset')

    ## model and dataset args
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help='bert model to fine-tune')
    parser.add_argument("--data_dir", default='', type=str, help='directory for all sentiment datasets')
    # parser.add_argument("--dataset", required=True, type=str, help='argumentation datasets to be used for training.')
    parser.add_argument("--output_dir", default='./', type=str, help='directory to store the model')
    # parser.add_argument("--logging_dir", default='', type=str, help='directory to store the logs')
    parser.add_argument("--num_labels", default=5, type=int)
    ## output_dir in training args to save the model
    # parser.add_argument('--save_model', action='store_true')
    # parser.add_argument("--save_model_dir", default='', type=str, help='directory to store the model')
    # parser.add_argument("--model_name", default='', type=str, help='name of the model inside save_dir')

    ## training args, for more details: https://huggingface.co/transformers/_modules/transformers/training_args.html#TrainingArguments
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    # parser.add_argument("--per_device_eval_batch_size", default=16, type=int)
    # parser.add_argument("--warmup_steps", default=500, type=int)
    # parser.add_argument("--weight_decay", default=0.01, type=int)
    parser.add_argument("--logging_steps", default=5, type=int)
    # parser.add_argument("--save_strategy", default='steps', type=str)
    parser.add_argument("--save_steps", default=0, type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument("--evaluation_strategy", default='epoch', type=str)
    # parser.add_argument("--evaluation_steps", default=100, type=int)

    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    tokenizer.add_special_tokens({'additional_special_tokens':['[EDU_SEP]']})

    ## training data
    edus_files, labels_files = os.path.join(args.data_dir, 'training_data/*.edus'), os.path.join(args.data_dir, 'training_data/*.labels')
    argdata = ArgumentDataset(tokenizer, edus_files, labels_files)

    ## validation_data
    val_edus_files, val_labels_files = os.path.join(args.data_dir, 'validation_data/*.edus'), os.path.join(args.data_dir, 'validation_data/*.labels')
    val_argdata = ArgumentDataset(tokenizer, val_edus_files, val_labels_files)

    ## edu_sep_id = tokenizer.convert_tokens_to_ids('[EDU_SEP]')
    config = BertConfig.from_pretrained(args.bert_model, num_labels=args.num_labels)
    edu_tag_model = BertForPhraseClassification.from_pretrained(args.bert_model, config=config)
    edu_tag_model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,      
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,  
        save_steps=args.save_steps, 
        do_train=args.do_train,
        do_eval=args.do_eval,
        # dataloader_drop_last=True,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
    )

    trainer = Trainer(
        model=edu_tag_model,        
        args=training_args,                
        train_dataset=argdata,
        eval_dataset=val_argdata,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=val_argdata)
    print(metrics)

if __name__ == '__main__':
    main()
