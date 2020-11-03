# _*_coding:utf-8_*_

# @Time : 2020/11/1 15:03
# @Author : xiaoyuge
# @File : build_freq.py 
# @Software: PyCharm

import pickle
from collections import Counter
import jieba
import pandas as pd
import numpy as np
import os


def cut(sentence):
    return ' '.join(jieba.cut(sentence))


raw_data = './data/sqlResult_1558435.csv'
stop_words = './data/stop_words.txt'
tokens_npy = './data/tokens.npy'
freq_pickle = './data/freq.pkl'


stopwords = [l.strip() for l in open(stop_words, 'r', encoding='utf-8', errors='ignore').readlines()]


# 读取新闻数据
def get_news(file_path, stopwords):
    if os.path.exists(tokens_npy):
        temp = np.load(tokens_npy)
    else:
        df = pd.read_csv(file_path, encoding='gb18030')
        content = df['content'].fillna('').apply(cut)
        tokens = [t for l in content.tolist() for t in l.split() if t not in stopwords]
        temp = np.array(tokens)
        np.save(tokens_npy, temp)
    return temp


def get_freq():
    tokens = get_news(raw_data, stopwords)
    if os.path.exists(freq_pickle):
        return pickle.load(open(freq_pickle, 'rb'))
    else:
        counter = Counter(tokens)
        total = sum(counter.values())
        freq = {w: count/total for w,count in counter.items()}
        with open(freq_pickle, 'wb') as f:
            pickle.dump(freq, f, protocol=pickle.DEFAULT_PROTOCOL)
        return freq


if __name__ == "__main__":
    freq = get_freq()
    print(freq)
