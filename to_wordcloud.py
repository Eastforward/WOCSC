#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Winter_Olympics_commentary_sentiment_classification 
@File ：to_wordcloud.py
@IDE  ：PyCharm 
@Author ：Eastforward
@Date ：2022/2/21 11:41 
"""
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

X, Y = 690, 690

neg_mask = np.array(Image.open("./wordcloud_img/bing.jpeg"))
pos_mask = np.array(Image.open("./wordcloud_img/xue.jpg"))

pos_comments = csv.reader(open('./data_analysis/positive.csv', 'r', encoding="utf-8"))
neg_comments = csv.reader(open('./data_analysis/negative.csv', 'r', encoding="utf-8"))
pos_comment = ""
neg_comment = ""
for comment in pos_comments:
    if not pos_comment:
        pos_comment += comment[0]
    else:
        pos_comment = " ".join([pos_comment, comment[0]])
for comment in neg_comments:
    if not neg_comment:
        neg_comment += comment[0]
    else:
        neg_comment = " ".join([neg_comment, comment[0]])

neg_wordcloud = WordCloud(background_color='white', mask=neg_mask, max_words=200, width=X, height=Y).generate(
    neg_comment)
pos_wordcloud = WordCloud(background_color='white', mask=pos_mask, max_words=200, width=X, height=Y).generate(
    pos_comment)

plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()
plt.savefig("wordcloud_img/negative.png")
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.show()
plt.savefig("wordcloud_img/positive.png")
