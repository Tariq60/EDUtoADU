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

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertPreTrainedModel
from transformers.utils import logging

from transformers.file_utils import ModelOutput
from typing import Optional

# logger = logging.get_logger(__name__)

# Trainer imports
from transformers import Trainer, TrainingArguments
from dataset import ArgumentDataset
from model import BertForPhraseClassification

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

paragraph_files, edus_files, labels_files = '../data/ets/para_text/*', '../data/ets/para_edu/*', '../data/ets/para_edu_label_all/*'
argdata = ArgumentDataset(tokenizer, paragraph_files, edus_files, labels_files)
edu_tag_model = BertForPhraseClassification.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./',      
    num_train_epochs=3,
    per_device_train_batch_size=16,  
    save_steps=0, 
    do_train=True,
    dataloader_drop_last=True
)

trainer = Trainer(
    model=edu_tag_model,        
    args=training_args,                
    train_dataset=argdata,
)

trainer.train()