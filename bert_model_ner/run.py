import json
from sklearn.model_selection import train_test_split
from bert_model_ner.model import BertNERModel, SegDataset, collate_fn
from torch.optim import Adam
from utils import config
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer


def get_label_id(train_ner_dir, feature_ner_model):
    data, label2id = [], {}
    with open(train_ner_dir, "r", encoding='utf-8') as fp:
        for line in fp.readlines():
            line = json.loads(line)
            data.append(line)
            for label in line['label']:
                if label not in label2id.keys():
                    label2id[label] = len(label2id)
    feature2id = {'O': 0}
    for i in range(len(label2id)):
        feature2id[f'B{i}'] = len(feature2id)
        feature2id[f'I{i}'] = len(feature2id)
    id2label = {v: k for k, v in label2id.items()}
    id2feature = {v: k for k, v in feature2id.items()}
    feature_label = {'labels2id': label2id, 'feature2id': feature2id, 'id2label': id2label,
                     'id2feature': id2feature}
    with open(feature_ner_model, 'w', encoding='utf-8') as fp:
        json.dump(feature_label, fp, ensure_ascii=False)
    return label2id, data, feature2id


def train():
    print("显卡信息：", config.device)
    label2id, data, feature2id = get_label_id(config.train_ner_dir, config.feature_ner_model)

    train_data, dev_data = train_test_split(data, test_size=0.1, random_state=2)
    # train_data[0]:{'query': '髯须粗壮，外表豪放，充满男人气概。', 'sentence': ['髯须粗壮，外表豪放，充满男人气概'], 'label': ['贾琏']}
    print(f"train_data: {len(train_data)}  dev_data: {len(dev_data)}")
    dataset_train = SegDataset(train_data, config.bert_model, label2id, feature2id)
    dataset_dev = SegDataset(dev_data, config.bert_model, label2id, feature2id)

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset_dev, batch_size=config.batch_size, shuffle=True,
                                collate_fn=collate_fn)

    model = BertNERModel.from_pretrained(config.bert_model, num_labels=len(feature2id))
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
        total_loss_train, total_acc_dev, total_loss_dev = 0, 0, 0
        model.train()
        for train_word, train_tag in tqdm(dataloader_train):
            train_tag = train_tag.to(config.device)
            input_ids = train_word['input_ids'].squeeze(1).to(config.device)
            attention_mask = train_word['attention_mask'].squeeze(1).to(config.device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=train_tag)
            model.zero_grad()  # 清零梯度
            total_loss_train += output[0].item()
            output[0].backward()  # 反向传播
            optimizer.step()  # 更新参数

        model.eval()
        with torch.no_grad():
            for dev_word, dev_tag in dataloader_dev:
                dev_tag = dev_tag.to(config.device)
                input_ids = dev_word['input_ids'].squeeze(1).to(config.device)
                attention_mask = dev_word['attention_mask'].squeeze(1).to(config.device)
                output = model(input_ids, attention_mask=attention_mask, labels=dev_tag)
                logits = model.crf.decode(output[1], mask=attention_mask.to(torch.bool))
                for k, v in zip(logits, dev_tag.tolist()):
                    if k == v[:len(k)]:
                        total_acc_dev += 1
                total_loss_dev += output[0].item()

        print(f"""
                  | epoch     num: {i + 1}
                  | Train loss f1: {total_loss_train / len(train_data):.3f}
                  | Dev   loss f1: {total_loss_dev / len(dev_data):.3f}
                  | Dev   acc f1: {total_acc_dev / len(dev_data):.3f}
              """)
        # 保存模型
        if (i == 5):
            model.save_pretrained(config.model_dir_ner)
            break


def test_on_sentence(query):
    model = BertNERModel.from_pretrained(config.model_dir_ner)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    data = tokenizer(query,
                     padding='longest',
                     max_length=512,
                     truncation=True,
                     return_tensors='pt')
    output = model(**data)
    logits = model.crf.decode(output, mask=data['attention_mask'].to(torch.bool))[0]

    with open(config.feature_ner_model, 'r', encoding="utf-8") as fp:
        target = json.load(fp)

    id2labels = {v: k for k, v in target['labels2id'].items()}
    id2feature = {v: k for k, v in target['feature2id'].items()}
    print(id2labels)
    print(id2feature)
    print(logits)
    sentence_id, label_id = [], []
    for i in range(len(logits)):
        if logits[i] == 0:
            sentence_id.append([])
        else:
            if not sentence_id[-1]:
                label_id.append(logits[i])
            sentence_id[-1].append(data['input_ids'][0][i].item())
    sentence_id = [i for i in sentence_id if i]

    res = []
    for i, token in enumerate(sentence_id):
        tmp = "".join(tokenizer.convert_ids_to_tokens(token))
        tag = id2labels[int(id2feature[label_id[i]][1:])]
        res.append((tmp, tag))

    return res


if __name__ == '__main__':
    # res = test_on_sentence(
    #     '容颜绝世美貌，慈祥的眼眸中流露出无尽慈悲，身着白色宽袍，头戴莲花宝冠。身形瘦小，毛色雪白，眼神灵动，额头上有一只鹅黄色的单角。相貌清秀，一袭白衣，带着一丝文士气息。')
    # for i in res:
    #     print(i)
    # train()
    local_rank = 0
    device = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.is_available())
