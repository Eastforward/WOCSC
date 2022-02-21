#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Winter_Olympics_commentary_sentiment_classification
@File ：comments_preprocessing.py
@IDE  ：PyCharm
@Author ：Eastforward
@Date ：2022/2/16 12:42
"""

import os
from get_video_id import get_youtube_video_id
import googleapiclient.discovery
import json
import csv


def main(videoId, path):
    print(videoId, path, '-' * 64)
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "你自己的"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=videoId
    )
    response = request.execute()

    with open(f"{path}/{videoId}.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(response))

    # for key,value in response.items():
    #     print(f"-------------------------\nkey: {key}\n\nvalue:{value}")

    # print(response)


def read_comment(file_name, path, video_caption):
    """
        考虑到楼中楼的回复有可能是针对层主，因此只统计主楼层的评论(即第一评论，不包含评论中的评论)
    """
    with open(f"{path}/{file_name}.json", encoding='utf-8') as load_f:
        headers = ['Video Caption', 'Comment']
        with open(f"{path}/{file_name}.csv", "w", encoding="utf-8", newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            load_dict = json.load(load_f)
            for item in load_dict["items"]:
                f_csv.writerow([video_caption, item['snippet']['topLevelComment']['snippet']['textOriginal']])


def preprocessing():
    """
        所有视频的标题和id
    """
    video_by_keywords = get_all_filepath("./videos/")
    print(video_by_keywords)
    for videos in video_by_keywords:
        try:
            os.mkdir(videos[:-4])
        except:
            pass
        # print(video[:-4])
        video = csv.reader(open(videos, 'r', encoding="utf-8"))
        next(video)  # 跳过表头
        for v in video:
            # 这里可能有多种情况，视频失效，评论关闭等等，直接tryexcept不想太多特判
            try:
                print(v, videos[:-4])
                main(v[1], videos[:-4])
                read_comment(v[1], videos[:-4], v[0])
            except:
                pass


def get_all_filepath(dir, types=".csv"):
    all_file_list = []
    parents = os.listdir(dir)
    for parent in parents:
        child = os.path.join(dir, parent)
        if os.path.isdir(child):
            get_all_filepath(child)
        else:
            suffix = os.path.splitext(child)[1]
            # print(suffix)
            if suffix == types:
                all_file_list.append(child)

    return all_file_list


if __name__ == "__main__":
    # main()
    # url = ['KuPuXAUXZjY','_fqQEYNBb7Y','LrIuNL0yKp0']
    # for u in url:
    #     main(u)
    #     read_comment(u)

    preprocessing()

    # AIzaSyBvbs_OPA6ggfarhM1iwq3BJgjiCcOpW-0
