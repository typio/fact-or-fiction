import os
import pandas as pd
from bs4 import BeautifulSoup

import requests
import concurrent.futures

data = pd.DataFrame(columns=['Label', 'Claim'])

PAGES = 300
FALSE = 0
TRUE = 1

urls = {
    "https://www.snopes.com/fact-check/rating/true/?pagenum=PAGE_INDEX": [TRUE, PAGES],
    "https://www.snopes.com/fact-check/rating/mostly-true/?pagenum=PAGE_INDEX": [TRUE, PAGES],
    "https://www.snopes.com/fact-check/rating/legit/?pagenum=PAGE_INDEX": [TRUE, PAGES],
    "https://www.snopes.com/fact-check/rating/correct-attribution/?pagenum=PAGE_INDEX": [TRUE, PAGES],

    "https://www.snopes.com/fact-check/rating/false/?pagenum=PAGE_INDEX": [FALSE, PAGES],
    "https://www.snopes.com/fact-check/rating/mostly-false/?pagenum=PAGE_INDEX": [FALSE, PAGES],
    "https://www.snopes.com/fact-check/rating/scam/?pagenum=PAGE_INDEX": [FALSE, PAGES],
    "https://www.snopes.com/fact-check/rating/misattributed/?pagenum=PAGE_INDEX": [FALSE, PAGES],
    "https://www.snopes.com/fact-check/rating/miscaptioned/?pagenum=PAGE_INDEX": [FALSE, PAGES],
    "https://www.snopes.com/fact-check/rating/unfounded/?pagenum=PAGE_INDEX": [FALSE, PAGES],

    "https://www.politifact.com/factchecks/list/?page=PAGE_INDEX&ruling=true": [TRUE, PAGES],
    "https://www.politifact.com/factchecks/list/?page=PAGE_INDEX&ruling=mostly-true": [TRUE, PAGES],

    "https://www.politifact.com/factchecks/list/?page=PAGE_INDEX&ruling=barely-true": [FALSE, PAGES],
    "https://www.politifact.com/factchecks/list/?page=PAGE_INDEX&ruling=false": [FALSE, PAGES],
    "https://www.politifact.com/factchecks/list/?page=PAGE_INDEX&ruling=pants-fire": [FALSE, PAGES],
}


def request_claim(url, label, i):
    url = url.replace("PAGE_INDEX", str(i))
    try:
        r = requests.get(url)
    except:
        print(f"Couldn't get {url}")
    else:
        soup = BeautifulSoup(r.text, features="lxml")

        texts = []

        # politifact
        for tag in soup.find_all(attrs={'class': 'm-statement__quote'}):
            texts.append(tag.get_text())

        # snopes
        for tag in soup.select('div.article_info_wrap'):
            texts.append(tag.find(attrs={'class': 'article_title'}).get_text())

        # clean up
        for i, text in enumerate(texts):
            texts[i] = text.replace("\n", " ")
            texts[i] = " ".join(text.split())

        global data
        newStuff = pd.DataFrame.from_dict({'Label': label, 'Claim': texts})
        data = pd.concat([data, newStuff])

        print(f'Successfully got {url}')


def write_file(str, label, i, set):
    folderpath = os.path.join('data', set)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath = os.path.join(folderpath, f"{label}")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(os.path.join(filepath, f"{i}.txt"), "w", encoding="utf-8") as f:
        f.write(str)


def build_data():
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as e:
        for url, params in urls.items():
            for i in range(params[1]):
                e.submit(request_claim, url, params[0], i)

    global data
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle
    data.to_csv('scrape_data.csv', index=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as e:
        for i, row in data.iterrows():
            set = 'train' if i < len(data)/2 else 'test'
            e.submit(write_file, row['Claim'], row['Label'], i, set)


build_data()
