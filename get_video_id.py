from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv


def get_youtube_video_id(keyword, scroll_times=1):
    """
    用于获取油管上的视频名称和video
    :param keyword: 搜索的关键字
    :param scroll_times: 页面滚动的次数
    :return: None
    """
    # 加上参数，禁止 chromedriver 日志写屏
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    wd = webdriver.Chrome("D:/tools/chromedriver.exe", chrome_options=options)

    wd.implicitly_wait(10)  # 如果找不到元素， 每隔 半秒钟 再去界面上查看一次， 直到找到该元素， 或者 过了10秒 最大时长。

    wd.get("http://www.youtube.com")
    time.sleep(2)  # 搜索中文的时候会慢一点
    ele = wd.find_elements(By.ID, 'search')
    ele[1].send_keys(keyword)
    but = wd.find_element(By.ID, 'search-icon-legacy')
    time.sleep(2)
    but.click()
    time.sleep(2) # 搜索中文的时候会慢一点

    for i in range(scroll_times):
        wd.execute_script('window.scrollBy(0,1000)')
        time.sleep(0.5)

    ele_video_id = wd.find_elements(By.CSS_SELECTOR, '.yt-simple-endpoint.style-scope.ytd-video-renderer')
    videos = {}
    for e in ele_video_id:
        if e.get_attribute('href') is not None:
            video_id = e.get_attribute('href')[-11:]
            video_name = e.find_elements(By.CSS_SELECTOR, '.style-scope.ytd-video-renderer')
            videos[video_name[1].text] = video_id

    headers = ['Video Caption', 'Video ID']

    with open(f"./videos/{keyword}.csv", "w", encoding="utf-8", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)

        for key, value in videos.items():
            print(64 * "-")
            print(f"caption: {key}")
            print(f"videoId: {value}")
            f_csv.writerow([key.strip(), value.strip()])

    print(64 * "-")
    # wd.quit()


if __name__ == '__main__':
    keywords = ["Olympic 2022", "BEIJING 2022", "Winter Olympic",
                "Beijing Winter Olympic", "北京冬奥村", "北京冬奥村餐厅",
                "冬奥vlog", "Winter Olympic Vlog", "Winter Olympic Village",
                "Winter Olympic Dining Hall"]
    scroll_times = 10
    for keyword in keywords:
        get_youtube_video_id(keyword, scroll_times)
