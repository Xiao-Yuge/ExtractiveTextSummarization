# _*_coding:utf-8_*_

# @Time : 2020/11/1 15:40
# @Author : xiaoyuge
# @File : model.py 
# @Software: PyCharm

import jieba
import networkx
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

stop_words = './data/stop_words.txt'


class SummaryModel:
    def __init__(self, freq, word_model, min_len=8, constraint=50):
        self.freq = freq
        self.model = word_model
        self.stopwords = [l.strip() for l in open(stop_words, 'r', encoding='utf-8', errors='ignore').readlines()]
        self.min_len = min_len
        self.constraint = constraint
        self.pat = '，。…！？,'

    def build_summarization_sample(self, text):
        sentences = self.split_sentence(text)
        ranking_sentence = self.get_correlations(sentences, text)
        summary_len = 0
        summary_set = set()
        for sentence in ranking_sentence:
            if summary_len < self.constraint:
                summary_len += len(sentence)
                summary_set.add(sentence)
            else:
                break
        # 将句子按原顺序排序
        summary = []
        for sentence in sentences:
            if sentence in summary_set:
                summary.append(sentence)
        return ','.join(summary)

    def build_text_rank_summary(self, text):
        sentences = self.split_sentence(text)
        sentence_graph = self.get_connect_graph(sentences)
        ranking_sentence = networkx.pagerank(sentence_graph)
        ranking_sentence = dict(sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)).keys()
        summary_len = 0
        summary_set = set()
        for sentence in ranking_sentence:
            if summary_len < self.constraint:
                summary_len += len(sentence)
                summary_set.add(sentence)
            else:
                break
        # 将句子按原顺序排序
        summary = []
        for sentence in sentences:
            if sentence in summary_set:
                summary.append(sentence)
        return ','.join(summary)

    def split_sentence(self, text):
        """
        将文本拆分为句子，合并长度过短的句子
        :param text: 输入文本
        :return: 文本分割后的句子
        """
        text = str(text).strip().replace('\r', '').replace('\n', '')
        sentences = re.split('[' + self.pat + ']', text)
        s_sentences = []
        num_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            if len(sentence) < self.min_len and i+1 < num_sentences:
                sentences[i+1] = sentence+sentences[i+1]
            elif sentence:
                s_sentences.append(sentence)
        return s_sentences

    def get_correlations(self, sentences, text):
        """
        获取句子与文本的相似度，以相似性降序返回
        :param sentences: 句子
        :param text: 文本
        :return: 相似度
        """
        text_embedding = self._sentence_embedding(text)
        similarity = {}
        for sentence in sentences:
            sentence_embedding = self._sentence_embedding(sentence)
            try:
                similarity[sentence] = cosine_similarity(text_embedding.reshape(1, -1), sentence_embedding.reshape(1, -1))[0][0]
            except:
                print(sentence)
        similarity = dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True))
        return list(similarity.keys())

    def _sentence_embedding(self, sentence):
        """
        生成句子向量
        :param sentence:
        :return:
        """
        alpha = 1e-4
        max_freq = max(self.freq.values())
        words = jieba.lcut(sentence)
        sentence_vec = np.zeros(self.model.wv.vector_size)
        for word in words:
            if word in self.model.wv.vocab and word not in self.stopwords:
                weights = alpha / (alpha + self.freq.get(word, max_freq))
                sentence_vec += weights * self.model.wv[word]
        sentence_vec /= len(words)
        return sentence_vec

    def get_connect_graph(self, sentences, window=3):
        """
        将句子构建成图
        :param sentences:
        :return:
        """
        sentence_graph = networkx.Graph()
        for i, t in enumerate(sentences):
            word_tuples = [(sentences[connect], t) for connect in range(i-window, i+window+1)
                           if connect>=0 and connect<len(sentences)]
            sentence_graph.add_edges_from(word_tuples)
        return sentence_graph

    def build_text_rank_summary_old(self, text):
        """
        旧版本，使用句子相似度构建图
        """
        sentences = self.split_sentence(text)
        matrix = self.get_correlation_matrix(sentences)
        sentence_graph = networkx.from_numpy_matrix(matrix)
        scores = sorted(networkx.pagerank(sentence_graph).items(), key=lambda x: x[1], reverse=True)
        summary_len = 0
        summary_set = set()
        for i, _ in scores:
            if summary_len < self.constraint:
                summary_len += len(sentences[i])
                summary_set.add(sentences[i])
            else:
                break
        # 将句子按原顺序排序
        summary = []
        for sentence in sentences:
            if sentence in summary_set:
                summary.append(sentence)
        return ','.join(summary)

    def get_correlation_matrix(self, sentences):
        """
        生成句子相关性的共现矩阵
        :param sentences:句子
        :return: 相似度共现矩阵
        """
        matrix = np.zeros((len(sentences), len(sentences)))
        for i, s_r in enumerate(sentences):
            sr_embedding = self._sentence_embedding(s_r)
            for j, s_c in enumerate(sentences):
                if i > j:
                    sc_embedding = self._sentence_embedding(s_c)
                    matrix[i][j] = cosine_similarity(sr_embedding.reshape(1, -1), sc_embedding.reshape((1, -1)))[0][0]
        return matrix


if __name__ == "__main__":
    from build_freq import get_freq
    from build_w2v import get_w2v_model
    model = SummaryModel(get_freq(), get_w2v_model(), constraint=150)
    text = """
据报道，英伟达计划以400亿美元收购英国芯片设计公司Arm。但这笔交易在中国面临着新的难题。据报道，Arm的中国合资公司首席执行官吴雄昂（Allen Wu）持有该合资公司17%的股份。根据公司注册文件，吴雄昂于去年11月接手了一家关键投资公司，目前控制着Arm中国六分之四的股东。

吴雄昂控制的两家公司已在深圳提起诉讼，指控Arm和其在合资公司的主要合伙人（私募股权公司厚朴投资）在6月份将其非法免职。目前，吴雄昂仍管理着Arm中国的日常运营工作，并掌管着该公司的印章，对公司业务仍具有合法控制权。这也使得吴雄昂成为英伟达收购交易的一个主要障碍。

一名熟悉Arm中国董事会的知情人士说，他认为这笔交易的成功率只有50%。另外两名知情人士透露，Arm的当前所有者日本的软银集团已任命软银在中国团队的主管Eric Chen负责协调吴雄昂的退出事宜。他们称，讨论的遣散费大约在1亿到2亿美元之间。知情人士说，9月份的时候，双方似乎已经快达成和解，吴雄昂和Eric Chen都已经分别告诉同事们，吴雄昂将在月底前离职。

但是吴雄昂持有的16.6%的股份价值一直是问题症结所在。吴雄昂认为，自2018年成立以来，Arm中国的价值已经增长五倍，至今天的500亿元人民币（75亿美元）。目前尚不清楚，吴雄昂是打算持有还是出售这些股份，以及是否有适合Arm、厚朴和中国政府的买家。双方仍在调停阶段，并且也有可能达成协议。但最近，吴雄昂又告诉同事说，他是否会离开Arm中国仍然不确定。熟悉软银和厚朴的知情人士则表示，谈判可能会一直拖延。

Arm中国负责Arm在中国地区的授权许可交易，同时也会开展一些研发工作。2018年，软银称，Arm中国贡献了该公司五分之一的总销售额。6月份，Arm中国的董事会以7-1的投票结果，决定罢免吴雄昂，原因是吴雄昂被指与他的Alphatecture投资基金存在利益冲突。

一名发言人称，吴雄昂从一开始就已经向董事会披露了该投资基金的存在。Arm中国的闹剧发生之际，正值Arm和英伟达准备向中国市场监管机构申请批准收购交易的关键时刻。向中国市场监管机构提交申请时，需要Arm中国的配合及Arm中国的数据。两名知情人士透露，Arm和英伟达尚未向监管机构提交任何文件，其中一人表示，背后的原因是他们难以获得合资企业的控制权。

两名熟悉Arm中国的知情人士表示，在Arm中国内部，吴雄昂已经组件了自己的安全团队，拒绝Arm或Arm中国董事会的任何代表进入该公司，并且Arm总部发送给员工的邮件也被系统过滤器屏蔽。

Arm、软银和英伟达均拒绝发表评论。吴雄昂未回复评论请求。Arm中国发言人称，不会对任何猜测发表评论，并且将对“任何散布谣言企图损害公司声誉的机构和个人”采取法律行动。该发言人还表示“一个稳定的Arm中国，符合所有人的最佳利益。”
"""
    summary = model.build_summarization_sample(text)
    print(summary)
    summary = model.build_text_rank_summary(text)
    print(summary)
