#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Winter_Olympics_commentary_sentiment_classification 
@File ：tokenizer.py
@IDE  ：PyCharm 
@Author ：Eastforward
@Date ：2022/2/16 12:43 
"""
import re
from keras.datasets import imdb
import os
from preprocessing import get_all_filepath
import csv
import json
import numpy as np
from train import MAX_LEN
from translate import connect as translation


def word2vec():
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

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    path = "./comments/"
    comments = os.listdir(path)
    vecs = []

    for comment in comments:
        print(comment)
        v = csv.reader(open(path + comment, 'r', encoding="utf-8"))
        for text in v:
            word_vec = []
            for word in text[1].split(' '):
                try:
                    if word == "<START>":
                        word_vec = []
                        word_vec.append(float(word_index["<START>"]))
                    else:
                        if word.lower() in word_index:
                            word_vec.append(float(word_index[word.lower()]))  # 前面已经加上了偏置量
                        else:
                            word_vec.append(float(word_index["<UNK>"]))  # 如果未知则未知
                except:
                    pass
            # print(word_vec)
            vecs.append(np.array(word_vec))
            # print(decode_review(word_vec))
    return np.array(vecs)


def comments_cleaning():
    """
    将爬到的评论进一步清洗，保证数据的合法性
    :return:
    """
    dirs = os.listdir("./videos")
    paths = []  # 包含所有需要遍历的评论文件相对地址

    for dir in dirs:
        # 有可能目录不合法，我也懒得判断了
        try:
            paths.append(get_all_filepath("./videos/" + dir + '/', ".csv"))
        except:
            pass

    actual_comment = 0
    total_comment = 0
    print(paths)
    for videos in paths:
        for video in videos:
            comment_processed = []
            v = csv.reader(open(video, 'r', encoding="utf-8"))
            next(v)  # 跳过表头
            for comment in v:
                total_comment += 1
                try:
                    comment[1] = re.sub(r'[^\w\s]', '', comment[1])  # 处理掉特殊字符，保留英文汉字数字
                    comment[1] = comment[1].replace('\n', '').replace('_', '')  # 还有可能有换行，下划线等
                    # print(comment[1].replace(' ','').encode('UTF-8').isalnum())

                    if not comment[1].replace(' ', '').encode('UTF-8').isalnum():  # 判断是不是纯英文，如果不是则需要翻译
                        if len(comment[1]) <= MAX_LEN:  # 防止太长，浪费钱
                            comment_en = translation(comment[1])
                        else:
                            comment_en = translation(comment[1][:MAX_LEN])
                        comment_en = json.loads(comment_en)
                        if comment_en["errorCode"] == '0':  # 翻译成功
                            # print(comment_en["translation"][0])
                            comment_processed.append(comment_en["translation"][0])
                    else:
                        # 纯英文或者数字，无需翻译
                        comment_en = comment[1]
                        comment_processed.append(comment_en)
                        # print("yes")
                except:
                    pass
            # 创建一个csv，保存在别的地方
            with open(f"./comments/{video[-15:-4]}.csv", "a+", encoding="utf-8", newline="") as f:
                f_csv = csv.writer(f)
                for comment in comment_processed:
                    actual_comment += 1
                    print(63 * '=' + str(actual_comment) + 63 * '=')
                    print(f"video id: {video[-15:-4]}\ncomment: {comment}")
                    f_csv.writerow([video[-15:-4], "<START> " + comment])
                print(128 * '=')

    print(64 * '-')
    print(f"total comments: {total_comment}\tactual_comments: {actual_comment}")
    print(64 * '-')
    return total_comment, actual_comment


if __name__ == '__main__':
    total_comment, actual_comment = comments_cleaning()
    word_vectors = word2vec()
