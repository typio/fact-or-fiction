import os
import pandas as pd
from bs4 import BeautifulSoup

import requests
import concurrent.futures

data = pd.DataFrame(columns=['Claim', 'True'])


def request_claim(url, i):
    url = url.replace("PAGE_INDEX", str(i))
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, features="lxml")

        # politifact
        for tag in soup.find_all(attrs={'class': 'm-statement__quote'}):
            print(tag.get_text())

        # snopes
        for tag in soup.select('div.article_info_wrap'):
            print(tag.find(attrs={'class': 'article_title'}).get_text())

        text = text.replace("\n", " ")
        text = " ".join(text.split())

        # TODO: add to data

    except:
        print(f"Couldn't get {url}")


def write_file(str, label, i):
    if not os.path.exists(f'train'):
        os.makedirs(f'train')

    filepath = os.path.join(f'train', f"{label}")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(os.path.join(filepath, f"{i}.txt"), "w", encoding="utf-8") as f:
        f.write(str)

def build_data():
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as e:
        for i in range(0, 1000):
            e.submit(
                request_claim, "https://www.snopes.com/fact-check/rating/true/?pagenum=PAGE_INDEX", i)
            e.submit(
                request_claim, "https://www.politifact.com/factchecks/list/?page=PAGE_INDEX&ruling=true", i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as e:
        for row in data:
            e.submit(
                write_file, row['Label'], i)

build_data()