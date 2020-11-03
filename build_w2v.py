# _*_coding:utf-8_*_

# @Time : 2020/11/3 14:58 
# @Author : xiaoyuge
# @File : build_w2v.py 
# @Software: PyCharm

from gensim.models.word2vec import Word2Vec, LineSentence
import multiprocessing
import os


def get_w2v_model(file='./data/corpus_cutted.txt', model_path='./data/news.w2v.model'):
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(LineSentence(file), window=5,
                         workers=multiprocessing.cpu_count(), size=128,
                         iter=10, min_count=1)
        model.save(model_path)
    return model
