#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Winter_Olympics_commentary_sentiment_classification 
@File ：predict.py
@IDE  ：PyCharm 
@Author ：Eastforward
@Date ：2022/2/16 12:45 
"""
from tokenizer import word2vec, comments_cleaning
from train import *
import csv
import numpy as np
import math


def decode(text):
    # text = "<START> Enjoy your time in Beijing"
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def result(output):
    output = F.softmax(output)
    index = output.argmax()
    return index


def predict():
    PATH = 'imdb model/model.pth'
    # total_comment, actual_comment = comments_cleaning()
    word_vectors = word2vec()
    x_pred = pad_sequences(word_vectors, maxlen=MAX_LEN, padding="post", truncating="post")
    # 对词向量做与训练的时候相同的处理即可
    x_pred = torch.from_numpy(x_pred).type(torch.LongTensor)
    x_pred = x_pred.to('cpu')
    best_model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to('cpu')
    best_model.load_state_dict(torch.load(PATH))
    best_model.eval()

    pos , neg = (0,0)
    for x in x_pred:
        print(64 * '-')
        sentence = decode(x.cpu().numpy())
        print(sentence)
        print(best_model(x.view(1, MAX_LEN)))
        res = result(best_model(x.view(1, MAX_LEN)))
        if res == 0:
            with open(f"./data_analysis/negative.csv", "a+", encoding="utf-8", newline="") as f:
                f_csv = csv.writer(f)
                f_csv.writerow(
                    [sentence.replace('<START>', '').replace('<UNK>', '').replace('<PAD>', '').replace('UNUSED',
                                                                                                       '').strip(),
                     0])
            neg +=1
            print(f"negative :{neg}")
        else:
            with open(f"./data_analysis/positive.csv", "a+", encoding="utf-8", newline="") as f:
                f_csv = csv.writer(f)
                f_csv.writerow(
                    [sentence.replace('<START>', '').replace('<UNK>', '').replace('<PAD>', '').replace('UNUSED',
                                                                                                       '').strip(),
                     1])
            pos += 1
            print(f"positive :{pos}")
    print(f"postive: {pos}/{pos+neg} = {pos/pos+neg}%")

if __name__ == '__main__':
    predict()
