#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Winter_Olympics_commentary_sentiment_classification 
@File ：main.py
@IDE  ：PyCharm 
@Author ：Eastforward
@Date ：2022/2/16 12:46 
"""
from get_video_id import get_youtube_video_id
from preprocessing import preprocessing
from tokenizer import comments_cleaning, word2vec
from train import load_data, start_train, evaluation
from predict import predict
import time

keywords = [ "北京冬奥村餐厅",
            "冬奥vlog", "Winter Olympic Vlog", "Winter Olympic Village",
            "Winter Olympic Dining Hall"]
# "Olympic 2022", "BEIJING 2022", "Winter Olympic",
#             "Beijing Winter Olympic","北京冬奥村",
scroll_times = 50

if __name__ == '__main__':
    # 开始爬虫
    print(64*'-'+"开始爬虫"+64*'-')
    for keyword in keywords:
        get_youtube_video_id(keyword, scroll_times)

    # 开始预处理评论
    print(64 * '-' + "开始预处理评论" + 64 * '-')
    time.sleep(2)
    preprocessing()

    # 开始分词
    print(64 * '-' + "开始分词" + 64 * '-')
    time.sleep(2)
    total_comment, actual_comment = comments_cleaning()
    word_vectors = word2vec()

    # 开始训练
    print(64 * '-' + "开始训练" + 64 * '-')
    time.sleep(2)
    train_loader, test_loader = load_data()
    start_train(train_loader, test_loader)
    evaluation(test_loader)

    # 开始预测并存储
    print(64 * '-' + "开始训练" + 64 * '-')
    time.sleep(2)
    predict()
