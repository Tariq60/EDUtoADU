import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# each training instance consists of a paragraph, edu splits and edu labels

class ArgumentDataset(Dataset):
    
    def __init__(self, tokenizer, paragraph_files, edus_files, labels_files, max_len=128, max_edu_seq=50):
        
        self.max_len, self.max_edu_seq = max_len, max_edu_seq
        self.tokenizer = tokenizer
        self.paragraphs = [''.join(open(file).readlines()) for file in glob.glob(paragraph_files)]
        self.edus = [open(file).readlines() for file in glob.glob(edus_files)]
        self.labels = [open(file).readlines() for file in glob.glob(labels_files)]
        self.label2id = {'B-claim': 1, 'I-claim': 2, 'B-premise': 3, 'I-premise': 4, 'O' : 0}
        
        ######
        filterout = [7, 24, 89, 231, 298, 348, 370, 373, 421, 473, 481, 485, 496, 508, 599, 680]
        for i in filterout[::-1]:
            self.paragraphs.pop(i); self.edus.pop(i); self.labels.pop(i)
        # print(len(self.paragraphs), len(self.edus), len(self.labels))
        ######
        
        self.labels = [
            [{'edu': line.rstrip().split('\t')[0], 'tokens': line.rstrip().split('\t')[1]} for line in para_labels]
                      for para_labels in self.labels
        ]
        self.edus_tokenized = [self.tokenizer(para_edus, truncation=True, padding='max_length', max_length=self.max_len) for para_edus in self.edus]       
        self.edus_tokenized2 = self.tokenizer.batch_encode_plus(self.edus[0], padding='max_length', max_length=self.max_len)
        
        self.edu_seq_input_ids = torch.full((len(self.edus), self.max_edu_seq, self.max_len), 0)
        self.edu_seq_attention_mask = torch.full((len(self.edus), self.max_edu_seq, self.max_len), 0)
        self.edu_seq_token_type_ids = torch.full((len(self.edus), self.max_edu_seq, self.max_len), 0)
        self.label_edus = [[0 for _ in range(self.max_edu_seq)] for _ in self.labels]
        self.label_tokens = [[[0 for _ in range(self.max_len)] for _ in range(self.max_edu_seq)] for _ in self.labels]
        
        for i, para_edus in enumerate(self.edus_tokenized):
            for j in range(min(self.max_edu_seq, len(para_edus['input_ids']))):
                self.edu_seq_input_ids[i][j] = torch.tensor(para_edus['input_ids'][j])
                self.edu_seq_attention_mask[i][j] = torch.tensor(para_edus['attention_mask'][j])
                self.edu_seq_token_type_ids[i][j] = torch.tensor(para_edus['token_type_ids'][j])
                
        for i, para_edus in enumerate(self.labels):
            for j in range(min(self.max_edu_seq, len(self.labels[i]))):
                self.label_edus[i][j] = self.label2id[self.labels[i][j]['edu']]
                for k in range(min(self.max_len, len(self.labels[i][j]['tokens'].split()))):
                    self.label_tokens[i][j][k] = self.label2id[self.labels[i][j]['tokens'].split()[k]]
        
        # self.paragraphs_tokenized = self.tokenizer(self.paragraphs, truncation=True, padding='max_length', max_length=512)
        # self.paragraphs_tokenized = [self.tokenizer.tokenize(p, truncation=True, padding='max_length', max_length=128) for p in self.paragraphs]
        # assert len(self.paragraphs) == len(self.edus) == len(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return { 'edu_seq_input_ids' : self.edu_seq_input_ids[i],
                'edu_seq_attention_mask': self.edu_seq_attention_mask[i],
                'edu_seq_token_type_ids': self.edu_seq_token_type_ids[i],
                'edu_labels' : self.label_edus[i],
                'token_labels' : self.label_tokens[i]
            
        }
        # return {'paragraph': self.paragraphs_tokenized[i], 'edus': self.edus_tokenized[i], 'labels': self.labels[i]}
        

