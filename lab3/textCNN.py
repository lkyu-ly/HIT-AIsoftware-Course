import argparse
import os
import os.path
import pickle as pkl
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

# from test import test_data


def read_data(file):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")
    texts, labels = [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            texts.append(text)
            labels.append(label)
    return texts, labels


def built_curpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    embedding = nn.Embedding(len(word_2_index), embedding_num)
    pkl.dump([word_2_index, embedding], open("data_pkl", "wb"))
    return word_2_index, embedding


class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][: self.max_len]
        label = int(self.all_label[index])
        text_idx = [self.word_2_index.get(i, 1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)
        return text_idx, label

    def __len__(self):
        return len(self.all_text)


class Block(nn.Module):
    def __init__(self, kernel_s, embeddin_num, max_len, hidden_num):
        super().__init__()
        # shape [batch * in_channel * max_len * emb_num]
        self.cnn = nn.Conv2d(
            in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embeddin_num)
        )
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):
        c = self.cnn(batch_emb)
        a = self.act(c)
        a = a.squeeze(dim=-1)
        m = self.mxp(a)
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)
        self.emb_matrix = emb_matrix
        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 2 * 3
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx):  # shape torch.Size([batch_size, 1, max_len])

        batch_emb = self.emb_matrix(
            batch_idx
        )  # shape torch.Size([batch_size,1, max_len, embedding])
        b1_result = self.block1(batch_emb)  # shape torch.Size([batch_size, 2])
        b2_result = self.block2(batch_emb)  # shape torch.Size([batch_size, 2])
        b3_result = self.block3(batch_emb)  # shape torch.Size([batch_size, 2])
        # 拼接
        feature = torch.cat(
            [b1_result, b2_result, b3_result], dim=1
        )  # shape torch.Size([batch_size, 6])
        pre = self.classifier(feature)  # shape torch.Size([batch_size,class_num])
        return pre


def test_data():
    args = parsers()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = pkl.load(open(args.data_pkl, "rb"))
    word_2_index, words_embedding = dataset[0], dataset[1]
    test_text, test_label = read_data(args.test_file)
    test_dataset = TextDataset(test_text, test_label, word_2_index, args.max_len)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    model = TextCNNModel(
        words_embedding, args.max_len, args.class_num, args.num_filters
    ).to(device)
    model.load_state_dict(torch.load(args.save_model_best))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_text, batch_label in test_dataloader:
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(label)
    accuracy = accuracy_score(all_true, all_pred)
    print(f"test dataset accuracy:{accuracy:.4f}")


def parsers():
    parser = argparse.ArgumentParser(description="TextCNN model of argparse")
    parser.add_argument(
        "--train_file", type=str, default=os.path.join("data", "train.txt")
    )
    parser.add_argument("--dev_file", type=str, default=os.path.join("data", "dev.txt"))
    parser.add_argument(
        "--test_file", type=str, default=os.path.join("data", "test.txt")
    )
    parser.add_argument(
        "--classification", type=str, default=os.path.join("data", "class.txt")
    )
    parser.add_argument(
        "--data_pkl", type=str, default=os.path.join("data", "dataset.pkl")
    )
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=38)
    parser.add_argument("--embedding_num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--num_filters", type=int, default=2, help="卷积产生的通道数")
    parser.add_argument(
        "--save_model_best", type=str, default=os.path.join("model", "best_model.pth")
    )
    parser.add_argument(
        "--save_model_last", type=str, default=os.path.join("model", "last_model.pth")
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if os.path.exists(args.data_pkl):
        dataset = pkl.load(open(args.data_pkl, "rb"))
        word_2_index, words_embedding = dataset[0], dataset[1]
    else:
        word_2_index, words_embedding = built_curpus(train_text, args.embedding_num)
    train_dataset = TextDataset(train_text, train_label, word_2_index, args.max_len)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, args.max_len)
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False)
    model = TextCNNModel(
        words_embedding, args.max_len, args.class_num, args.num_filters
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = nn.CrossEntropyLoss()
    acc_max = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0, 0
        for batch_index, (batch_text, batch_label) in enumerate(train_loader):
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)
            loss = loss_fn(pred, batch_label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss
            count += 1

            if (
                len(train_loader) - batch_index <= len(train_loader) % 1000
                and count == len(train_loader) % 1000
            ):
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0
            if batch_index % 1000 == 999:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))

                loss_sum, count = 0.0, 0
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch_text, batch_label in dev_loader:
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                pred = model(batch_text)
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()
                all_pred.extend(pred)
                all_true.extend(label)
        acc = accuracy_score(all_pred, all_true)
        print(f"dev acc:{acc:.4f}")
        if acc > acc_max:
            acc_max = acc
            torch.save(model.state_dict(), args.save_model_best)
            print(f"已保存最佳模型")
        print("*" * 50)
    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
    test_data()
