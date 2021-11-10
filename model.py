''' this model should use BertModel to extract the embeddings of the paragraph
    then do the following:
        1. use EDU split to get the embeddings of each tokens of an EDU
        2. represent the EDU as the average embedding of its member tokens
        3. pass the EDU embedding to the classifier layer to make predictions
        4. calculate the loss based on the predicted and gold EDU labels
'''
from collections import Counter

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, HingeEmbeddingLoss

from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertPreTrainedModel



class BertForPhraseClassification(BertPreTrainedModel):

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, edu_sequence_length=50):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.edu_sep_id = 30522
        self.edu_sequence_length = edu_sequence_length

        self.bert = BertModel(config, add_pooling_layer=False)
        # self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # print(self.config)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        token_labels=None,
    ):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # for batch in sequence_output:
        #     for t in batch:
        #         print('not in function', t[0])
        # print(input_ids.shape, sequence_output.shape)
        edu_embeddings = self.get_edu_emb(input_ids, sequence_output)
        # print(edu_embeddings.shape)
        # print(edu_embeddings[-1][:10])
        # if any(torch.isnan(edu_embeddings).view(-1)):
        #     print('This edu_embeddings Tensor has a NaN!!!!!!')
        
        edu_embeddings = self.dropout(edu_embeddings)
        logits = self.classifier(edu_embeddings)
        # print(logits.shape, labels.shape)
        # print(logits.view(-1, self.num_labels).shape, labels.view(-1).shape)
        # print(logits.view(-1, self.num_labels)[:10], labels.view(-1)[:10])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        # output = (logits,) + outputs[2:]
        # return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    def get_edu_emb(self, input_ids, outputs, edu_seperator_id=30522, max_edu_len=10):
        'Returns a sequence of 50 EDUs (padded or truncated) per paragraph represented as the average embeddings of their tokens'
        batch_size = outputs.shape[0]
        batch_para_edu_avg_emb = torch.zeros(batch_size,  self.edu_sequence_length, self.config.hidden_size)
        
        # finding the "[EDU_SEP]" token in each paragraph given a batch of input_ids
        # seperators[0] has index of the paragraph in a batch
        # seperators[1] has index of "[EDU_SEP]" in all paragraphs
        seperators = (input_ids == edu_seperator_id).nonzero(as_tuple=True)
        # print(seperators[0])
        # print(seperators[1])
        
        # getting the number of edus in each paragraph
        edu_per_para, all_keys, i = [], range(batch_size), 0
        for k, v in Counter([t.item() for t in seperators[0]]).items():
            while k != all_keys[i] and i < len(all_keys):
                edu_per_para.append(0); i+=1
            edu_per_para.append(v); i+=1

        # print(edu_per_para)
        
        # calculating the average embeddings for each EDU
        seperators_idx = 0
        for i, edu_count_per_para in enumerate(edu_per_para):
            prev_edu_sep, j = 0, 0
            for j in range(edu_count_per_para):
                if j < self.edu_sequence_length:
                    cur_edu_sep = seperators[1][seperators_idx].item()
                    # print(i, j, prev_edu_sep, cur_edu_sep)
                    assert input_ids[i][prev_edu_sep] in [101, edu_seperator_id]
                    assert input_ids[i][cur_edu_sep] in [edu_seperator_id, 102]

                    # print(i, j, len(outputs[i][prev_edu_sep+1:cur_edu_sep]))
                    # print(prev_edu_sep, cur_edu_sep)
                    # cur_edu_sep_max_len = min(cur_edu_sep, max_edu_len)
                    if cur_edu_sep - prev_edu_sep > 1:
                        batch_para_edu_avg_emb[i][j] = torch.mean(outputs[i][prev_edu_sep+1:cur_edu_sep], dim=0)
                    # if any(torch.isnan(batch_para_edu_avg_emb[i][j]).view(-1)):
                    #     print(batch_para_edu_avg_emb[i][j])
                    #     print(input_ids[i])
                    # for t_i, t in enumerate(outputs[i][prev_edu_sep+1:cur_edu_sep]):
                    #     if any(torch.isnan(t).view(-1)):
                    #         print(t_i, t)
                    prev_edu_sep = cur_edu_sep

                seperators_idx += 1;
            
            if j+1 < self.edu_sequence_length:
                # calculating embeddings of the last EDU that is between [EDU_SEP] and [SEP]
                cur_edu_sep = (input_ids[i] == 102).nonzero(as_tuple=True)[0].item()
                # print(i, j+1, prev_edu_sep, cur_edu_sep)
                assert input_ids[i][prev_edu_sep] in [101, edu_seperator_id]
                assert input_ids[i][cur_edu_sep] in [edu_seperator_id, 102]

                # print('final edu check:', i, j+1, len(outputs[i][prev_edu_sep+1:cur_edu_sep]))
                # print(prev_edu_sep, cur_edu_sep)
                # cur_edu_sep_max_len = min(cur_edu_sep, max_edu_len)
                if cur_edu_sep - prev_edu_sep > 1:
                    batch_para_edu_avg_emb[i][j+1] = torch.mean(outputs[i][prev_edu_sep+1:cur_edu_sep], dim=0)
                # if any(torch.isnan(batch_para_edu_avg_emb[i][j+1]).view(-1)):
                #         print(batch_para_edu_avg_emb[i][j+1])
                #         print(input_ids[i])
                # for t_i, t in enumerate(outputs[i][prev_edu_sep+1:cur_edu_sep]):
                #     if any(torch.isnan(t).view(-1)):
                #         print(t_i, t)
            
        return batch_para_edu_avg_emb.to('cuda:0')


        