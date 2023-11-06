import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
import json


class BertClassifierModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifierModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.entropy_loss = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                           attention_mask=attention_mask)[1]
        pooled_output = self.dropout(output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.entropy_loss(logits, labels)
            return loss, logits
        return logits


class SegDataset(Dataset):
    def __init__(self, word, label, bert_model, label2id):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.labels = [label2id[i] for i in label]
        self.words = [self.tokenizer(w,
                                     padding='longest',
                                     max_length=512,
                                     truncation=True,
                                     return_tensors='pt') for w in word]

    def __getitem__(self, item):
        word = self.words[item]
        label = self.labels[item]
        return word, label

    def __len__(self):
        return len(self.labels)






