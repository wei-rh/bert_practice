import json
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import BertClassifierModel, SegDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from utils import config


def get_label_id(configs):
    labels = []
    with open(configs.train_dir, "r", encoding='utf-8') as fp:
        for line in fp.readlines():
            label, word = line.split("：")
            labels.append(label.strip())

    id2label = {k: v for k, v in enumerate(set(labels))}
    label2id = {v: k for k, v in id2label.items()}
    with open(configs.feature_class_model, 'w', encoding="utf-8") as fp:
        json.dump(id2label, fp, ensure_ascii=False)
    return id2label, label2id


def collate_fn(batch):
    max_length = max([len(i[0]['input_ids'][0]) for i in batch])
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    for item in batch:
        pad_length = max_length - len(item[0]['input_ids'][0])
        input_ids.append(F.pad(item[0]['input_ids'], (0, pad_length), value=0).tolist())
        attention_mask.append(F.pad(item[0]['attention_mask'], (0, pad_length), value=0).tolist())
        token_type_ids.append(F.pad(item[0]['token_type_ids'], (0, pad_length), value=0).tolist())
        labels.append(item[1])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    return data, labels


def train():
    print("显卡信息：", config.device)
    words, labels = [], []
    with open(config.train_dir, "r", encoding='utf-8') as fp:
        for line in fp.readlines():
            label, word = line.split("：")
            words.append(word.strip())
            labels.append(label.strip())

    id2label, label2id = get_label_id(config)
    train_data, dev_data, train_label, dev_label = train_test_split(words, labels, test_size=0.1, random_state=2)
    print(f"train_data: {len(train_data)}  dev_data: {len(dev_data)}")

    dataset_train = SegDataset(train_data, train_label, config.bert_model, label2id)
    dataset_dev = SegDataset(dev_data, dev_label, config.bert_model, label2id)

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True,
                                collate_fn=collate_fn)

    model = BertClassifierModel.from_pretrained(config.bert_model, num_labels=len(label2id))
    model.to(config.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.eps)

    for i in range(config.epoch):
        total_acc_train, total_loss_train, total_acc_dev, total_loss_dev = 0, 0, 0, 0
        model.train()
        for train_word, train_tag in tqdm(dataloader_train):
            train_tag = train_tag.to(config.device)
            input_ids = train_word['input_ids'].squeeze(1).to(config.device)
            attention_mask = train_word['attention_mask'].squeeze(1).to(config.device)
            output = model(input_ids, attention_mask=attention_mask, labels=train_tag)
            model.zero_grad()  # 清零梯度
            total_loss_train += output[0].item()
            total_acc_train += (output[1].argmax(dim=1) == train_tag).sum().item()
            output[0].backward()  # 反向传播
            optimizer.step()  # 更新参数

        model.eval()
        with torch.no_grad():
            for dev_word, dev_tag in dataloader_dev:
                dev_tag = dev_tag.to(config.device)
                input_ids = dev_word['input_ids'].squeeze(1).to(config.device)
                attention_mask = dev_word['attention_mask'].squeeze(1).to(config.device)
                output = model(input_ids, attention_mask=attention_mask, labels=dev_tag)
                total_acc_dev += (output[1].argmax(dim=1) == dev_tag).sum().item()
                total_loss_dev += output[0].item()
        train_f1 = total_acc_train / len(train_data)
        dev_f1 = total_acc_dev / len(dev_data)
        print(f"""
                  | epoch     num: {i + 1}
                  | Train loss f1: {total_loss_train / len(train_data):.3f}
                  | Train acc f1: {train_f1:.3f}
                  | Dev   loss f1: {total_loss_dev / len(dev_data):.3f}
                  | Dev   acc f1: {dev_f1:.3f}
              """)
        # 保存模型
        if (train_f1 == dev_f1 == 1) or (i == config.epoch - 1):
            model.save_pretrained(config.model_dir)
            break


def test_on_sentence(query):
    model = BertClassifierModel.from_pretrained(config.model_dir)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    data = tokenizer(query,
                     padding='longest',
                     max_length=512,
                     truncation=True,
                     return_tensors='pt')
    output = model(**data)
    with open(config.feature_class_model, 'r', encoding="utf-8") as fp:
        id2label = json.load(fp)
    print(id2label)
    return id2label[str(output.argmax(dim=1).item())]


if __name__ == '__main__':
    train()
    t = test_on_sentence('相貌俊美，仪态万方，智勇双全。')
    print(t)
