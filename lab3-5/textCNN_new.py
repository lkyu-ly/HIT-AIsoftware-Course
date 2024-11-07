import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 数据集类，用于封装数据和标签
class Data(Dataset):
    def __init__(self, x, y):
        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 字符串清理函数，用于预处理文本数据
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 加载数据和标签的函数
def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = list(
        open(positive_data_file, "r", encoding="utf-8").readlines()
    )
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(negative_data_file, "r", encoding="utf-8").readlines()
    )
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = list(map(lambda x: x.split(), x_text))

    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.array(positive_labels + negative_labels)
    return x_text, y


# 数据加载时的批处理函数
def collate_fn(batch):
    data, label = zip(*batch)
    return list(data), torch.LongTensor(label)


# 初始化ELMO模型的函数
def init_elmo():
    elmo = Elmo(elmo_options_file, elmo_weight_file, 1)
    for param in elmo.parameters():
        param.requires_grad = False
    return elmo


# 获取ELMO嵌入的函数
def get_elmo(model, sentence_lists):
    character_ids = batch_to_ids(sentence_lists)
    embeddings = model(character_ids)
    return embeddings["elmo_representations"][0]


# CNN块类，用于特征提取
class Block(nn.Module):
    def __init__(self, kernel_s, embeddin_num, hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embeddin_num)
        )
        self.act = nn.ReLU()

    def forward(self, batch_emb):
        c = self.cnn(batch_emb)
        a = self.act(c)
        a = a.squeeze(dim=-1)
        m = a.mean(dim=2)  # 改为对时间维度取平均
        return m


# 文本分类模型类，整合ELMO和CNN块
class TextCNNModel(nn.Module):
    def __init__(self, elmo, max_len, class_num, hidden_num):
        super().__init__()
        self.elmo = elmo
        self.block1 = Block(2, elmo_dim, hidden_num)
        self.block2 = Block(3, elmo_dim, hidden_num)
        self.block3 = Block(4, elmo_dim, hidden_num)
        self.classifier = nn.Linear(hidden_num * 3, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, sentence_lists):
        elmo_embeddings = get_elmo(self.elmo, sentence_lists)
        elmo_embeddings = torch.unsqueeze(elmo_embeddings, dim=1)
        b1_result = self.block1(elmo_embeddings)
        b2_result = self.block2(elmo_embeddings)
        b3_result = self.block3(elmo_embeddings)

        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)
        pre = self.classifier(feature)
        return pre


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Elmo配置文件路径
elmo_options_file = "elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
elmo_weight_file = "elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
elmo_dim = 512

elmo_model = init_elmo()

# 数据加载
x_text, y = load_data_and_labels("data/rt-polarity.pos", "data/rt-polarity.neg")
x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.2)

train_data = Data(x_train, y_train)
test_data = Data(x_test, y_test)
train_dataloader = DataLoader(
    train_data, batch_size=32, shuffle=True, collate_fn=collate_fn
)
test_dataloader = DataLoader(
    test_data, batch_size=32, shuffle=False, collate_fn=collate_fn
)

# 初始化模型
model = TextCNNModel(elmo_model, max_len=50, class_num=2, hidden_num=100).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 训练和评估模型的主循环，共进行10个epoch
for epoch in range(10):
    # 设置模型为训练模式
    model.train()
    train_loss = 0.0
    # 使用tqdm显示训练进度条
    train_loader_tqdm = tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{10} [Training]", leave=False
    )

    # 遍历训练数据集的每个批次
    for batch_data in train_loader_tqdm:
        batch_text, batch_label = batch_data
        batch_label = batch_label.to(device)

        pred = model(batch_text)
        loss = loss_fn(pred, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条描述
        train_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=train_loss / len(train_loader_tqdm))

    # 评估阶段
    model.eval()
    all_pred, all_true = [], []
    eval_loss = 0.0
    # 使用tqdm显示评估进度条
    eval_loader_tqdm = tqdm(
        test_dataloader, desc=f"Epoch {epoch+1}/{10} [Evaluating]", leave=False
    )

    with torch.no_grad():
        # 遍历测试数据集的每个批次
        for batch_data in eval_loader_tqdm:
            batch_text, batch_label = batch_data
            pred = model(batch_text)
            pred = torch.argmax(pred, dim=1)

            all_pred.extend(pred.cpu().numpy())
            all_true.extend(batch_label.cpu().numpy())

        acc = accuracy_score(all_true, all_pred)
        eval_loader_tqdm.set_postfix(accuracy=acc)

    tqdm.write(
        f"Epoch {epoch + 1}, Training Loss: {train_loss/len(train_loader_tqdm):.4f}, Validation Accuracy: {acc:.4f}"
    )
