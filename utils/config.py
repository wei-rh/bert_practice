import torch
from transformers import AutoTokenizer, AutoModel
bert_model = "../pretraining_model/bert_base_chinese/"
model_dir = "../save/classifier/"
model_dir_ner = "../save/ner/"
train_dir = "../processed_data/train.txt"
train_ner_dir = "../processed_data/train_ner.txt"
feature_class_model = "../processed_data/feature_class_model.json"
feature_ner_model = "../processed_data/feature_ner_model.json"
local_rank = 0
device = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
batch_size = 8
# 优化器的超参数
learning_rate = 2e-5  # 学习率
weight_decay = 1e-2  # 权重衰减，可选
eps = 1e-8  # Adam优化器中的epsilon参数，可选
epoch = 20

if __name__ == '__main__':
    model_dir = r"D:\chatglm_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).half().cuda()
    # print(torch.cuda.is_available())CUDA_VISIBLE_DEVICES