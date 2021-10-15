''' this model should use BertModel to extract the embeddings of the paragraph
    then do the following:
        1. use EDU split to get the embeddings of each tokens of an EDU
        2. represent the EDU as the average embedding of its member tokens
        3. pass the EDU embedding to the classifier layer to make predictions
        4. calculate the loss based on the predicted and gold EDU labels
'''
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, HingeEmbeddingLoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertPreTrainedModel

class BertForPhraseClassification(BertPreTrainedModel):

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, edu_sequence_length=50):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.edu_sequence_length = edu_sequence_length

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.edu_sequence_length, config.num_labels)

        self.init_weights()


    def forward(
        self,
        edu_seq_input_ids=None,
        edu_seq_attention_mask=None,
        edu_seq_token_type_ids=None,
        edu_labels=None,
        token_labels=None,
    ):
        # edu_outputs of size: batch_size(=16), msx edus in one paragraph (=50), and bert hidden layer size (=768)
        edu_outputs = torch.zeros(edu_seq_input_ids.shape[0], self.edu_sequence_length, self.config.hidden_size)
        for i in range(self.edu_sequence_length):
            outputs = self.bert(edu_seq_input_ids[:, i, :], attention_mask=edu_seq_attention_mask[:, i, :], token_type_ids=edu_seq_token_type_ids[:, i, :])
            print(outputs[1].shape, edu_outputs[i].shape, edu_outputs.shape)
            edu_outputs[:, i, :] = outputs[1]

        # outputs = self.bert(edu_seq_input_ids, attention_mask=edu_seq_attention_mask, token_type_ids=edu_seq_token_type_ids)
        # outputs = outputs[1]
        
        edu_outputs = self.dropout(edu_outputs)
        logits = self.classifier(edu_outputs)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output