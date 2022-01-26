import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# each training instance consists of a paragraph, edu splits and edu labels

class ArgumentDataset(Dataset):
    
    def __init__(self, tokenizer, edus_files, labels_files, max_len=256, max_edu_seq=50):
        
        self.max_len, self.max_edu_seq = max_len, max_edu_seq
        self.tokenizer = tokenizer
        
        # self.paragraphs = [''.join(open(file).readlines()) for file in glob.glob(paragraph_files)]
        self.edus = [open(file).readlines() for file in sorted(glob.glob(edus_files))]
        self.labels = [open(file).readlines() for file in sorted(glob.glob(labels_files))]
        self.label2id = {'B-claim': 1, 'I-claim': 2, 'B-premise': 3, 'I-premise': 4, 'O' : 0}
        
        assert len(self.labels) == len(self.edus)
        
        self.labels = [
            [{'edu': line.rstrip().split('\t')[0], 'tokens': line.rstrip().split('\t')[1]} for line in para_labels]
                      for para_labels in self.labels
        ]
        self.label_edus = [[0 for _ in range(self.max_edu_seq)] for _ in self.labels]
        self.label_tokens = [[[-100 for _ in range(self.max_len)] for _ in range(self.max_edu_seq)] for _ in self.labels]
        
        self.para_edu_splits = [' [EDU_SEP] '.join([line.rstrip() for line in para_edus]) for para_edus in self.edus]
        self.para_edu_splits_tok = self.tokenizer(self.para_edu_splits, truncation=True, padding='max_length', max_length=self.max_len)
                
        for i, para_edus in enumerate(self.labels):
            for j in range(min(self.max_edu_seq, len(self.labels[i]))):
                self.label_edus[i][j] = self.label2id[self.labels[i][j]['edu']]
                for k in range(min(self.max_len, len(self.labels[i][j]['tokens'].split()))):
                    self.label_tokens[i][j][k] = self.label2id[self.labels[i][j]['tokens'].split()[k]]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return {'input_ids': self.para_edu_splits_tok['input_ids'][i],
                'attention_mask': self.para_edu_splits_tok['attention_mask'][i],
                'token_type_ids': self.para_edu_splits_tok['token_type_ids'][i],
                'labels' : self.label_edus[i],
                'token_labels' : self.label_tokens[i]
               }
        