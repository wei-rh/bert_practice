import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torchcrf import CRF
import json


class BertNERModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNERModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                           attention_mask=attention_mask)[0]
        pooled_output = self.dropout(output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            crf_loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.to(torch.bool))
            loss = -1 * crf_loss
            return loss, logits

        return logits


class SegDataset(Dataset):
    def __init__(self, data, bert_model, label2id, feature2id):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.label2id = label2id
        self.feature2id = feature2id
        self.words, self.labels = self.processed(data)

    def processed(self, data):
        # train_data[0]:{'query': '髯须粗壮，外表豪放，充满男人气概。', 'sentence': ['髯须粗壮，外表豪放，充满男人气概'], 'label': ['贾琏']}
        words, labels = [], []
        for item in data:
            query = self.tokenizer(item['query'],
                                   padding='longest',
                                   max_length=512,
                                   truncation=True,
                                   return_tensors='pt')
            sentence = [self.tokenizer(w,
                                       padding='longest',
                                       max_length=512,
                                       truncation=True,
                                       return_tensors='pt') for w in item['sentence']]
            label = [self.label2id[i] for i in item['label']]

            tag = ['O' for _ in range(len(query['input_ids'][0]))]

            for n, sen in enumerate(sentence):
                sen_input_ids = sen['input_ids'][0][1:-1].tolist()
                for i in range(len(tag)):
                    query_input_ids = query['input_ids'][0].tolist()
                    if query_input_ids[i: i + len(sen_input_ids)] == sen_input_ids:
                        tag[i] = 'B' + str(label[n])
                        tag[i + 1: i + len(sen_input_ids)] = ['I' + str(label[n]) for _ in
                                                              range(len(sen_input_ids) - 1)]
                        break
            tag = [self.feature2id[i] for i in tag]
            words.append(query)
            labels.append(tag)
        return words, labels

    def __getitem__(self, item):
        word = self.words[item]
        label = self.labels[item]
        return word, label

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    max_length = max([len(i[0]['input_ids'][0]) for i in batch])
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    for item in batch:
        pad_length = max_length - len(item[0]['input_ids'][0])
        input_ids.append(F.pad(item[0]['input_ids'], (0, pad_length), value=0).tolist())
        attention_mask.append(F.pad(item[0]['attention_mask'], (0, pad_length), value=0).tolist())
        token_type_ids.append(F.pad(item[0]['token_type_ids'], (0, pad_length), value=0).tolist())
        labels.append(item[1] + [0 for _ in range(pad_length)])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    return data, labels


def get_label_id(config):
    labels = []
    with open(config.train_dir, "r", encoding='utf-8') as fp:
        for line in fp.readlines():
            label, word = line.split("：")
            labels.append(label.strip())

    id2label = {k: v for k, v in enumerate(set(labels))}
    label2id = {v: k for k, v in id2label.items()}
    with open(config.feature_class_model, 'w', encoding="utf-8") as fp:
        json.dump(id2label, fp, ensure_ascii=False)
    return id2label, label2id
