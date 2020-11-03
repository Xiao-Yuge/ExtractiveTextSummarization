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


if __name__ == "__main__":
    from build_freq import get_freq
    from build_w2v import get_w2v_model
    model = SummaryModel(get_freq(), get_w2v_model(), constraint=300)
    text = """
自阿里巴巴创始人马云在外滩金融峰会上语惊四座，直言银行是当铺思想，巴塞尔协议是“老人俱乐部”，在金融行业引发巨大争议后，关于大型科技公司的创新风险及后续监管思路一直是热议话题。如今，对于大型科技企业创新监管框架，答案或已在酝酿。据证监会发布，11月2日，央行、银保监会、证监会、国家外汇管理局四部门联合对蚂蚁集团实际控制人马云、董事长井贤栋、总裁胡晓明进行监管约谈。同一天，央行行长易纲也对此再度重磅发声。后续如何在技术带来的便利与风险中让金融服务更安全，将是央行高度关注的重要命题。

　　频繁预警

　　争议之下，监管出手。11月2日，蚂蚁集团向北京商报记者确认，蚂蚁集团实际控制人与相关管理层接受了各主要监管部门的监管约谈。蚂蚁集团会深入落实约谈意见，继续沿着“稳妥创新、拥抱监管、服务实体、开放共赢”的十六字指导方针，继续提升普惠服务能力，助力经济和民生发展。

　　另据21财经报道，同一天，在香港金融科技周“数字经济中的央行角色”主题会议上，央行行长易纲也表示，大科技公司显著提高了金融服务水平，尤其是偏远地区的服务需求有所改善，如移动支付和二维码等技术的普及已经改变了游戏规则。不过，提升效率、降低成本的另一面，易纲也强调，商业秘密的保护和消费者个人隐私的保护，是极大的挑战。

　　针对易纲的最新发言，麻袋研究院高级研究员苏筱芮认为，主要释放了对大科技公司重点关注的信号，此后或将对大科技公司方面加强监管。在她看来，大型科技公司发挥的作用需要辩证看待：一方面，大科技公司创新能力强、科技基因根深蒂固，确实能够对传统金融产生助益；但另一方面，大科技公司在金融业务中牵扯过多，容易反客为主产生监管套利，这不仅仅是国内，更是一个国际化、全球化的议题。

　　“监管肯定了金融科技创新在提升金融服务效率方面的作用，但也需要防范其中的风险。未来需要平衡好金融安全与金融创新。”光大银行(3.980, 0.10, 2.58%)金融市场部分析师周茂华称。

　　风险犹在

　　近年来，金融科技发展势如破竹。但在这一过程中，大型科技公司在资源配置中形成市场垄断、以科技之名行金融之实，甚至出现数据泄露与侵权等一系列风险，引发关注。资深学者周矍铄称，当前，大型互联网企业进入金融领域，凭借技术优势掌握大量数据，辅以互联网技术的外部特征，容易形成市场主导地位，且在资源配置中权力过度集中，并逐步强化为市场垄断。

　　众所周知，金融服务必须满足特定资质要求，坚持持牌经营原则，严格准入和业务监督管理。周矍铄直言道，若大型互联网企业大量开展金融业务，但却宣称自己是科技公司，不仅是逃避监管，更容易无序扩张，造成风险隐患，不利于公平竞争，也不利于消费者保护。除了产品和业务边界模糊外，数据泄露与侵权风险同样不可忽视。

　　此外，目前业内也不乏以科技之名行金融之实的案例。苏筱芮称，当下，大型金融科技公司主要有两类风险，一是金融方面的风险，以科技之名行金融业务之实，容易积聚底层风险；二是技术方面的风险，例如信息安全与客户相关的权益保护等。“个人认为监管当下的策略没有问题，应先制定金控管理办法，把大科技公司先圈起来，之后何时管、怎样管可从长计议，监管是一个与时俱进的动态过程。”

　　券商投行资深人士何南野则认为，因为业务复杂，大型科技公司也容易导致风险集聚和外溢风险，以及创新业务不合法律等合规风险。从目前看，风险外溢及合规性问题都很严重，都需要金融科技公司和监管部门重点关注。

　　加强监管

　　业内多数认为，当前，大科技公司已经大到不能忽视，金融监管也必须跟上步伐。正如易纲所称，大数据、人工智能等新技术不仅给传统商业银行带来压力，也给央行带来新的挑战，如何在技术带来的便利与风险中让金融服务更安全，对央行来说是一个重要的命题。

　　11月2日上午，银保监会党委书记、主席郭树清主持召开党委（扩大）会议时也提到着力完善现代金融监管体系。处理好金融发展、金融稳定和金融安全的关系，提升金融监管能力。加强制度建设，坚持市场化、法治化、国际化原则，提高监管透明度。完善风险全覆盖的监管框架，增强监管的穿透性、统一性和权威性。依法将金融活动全面纳入监管，对同类业务、同类主体一视同仁。对各类违法违规行为“零容忍”，切实保护金融消费者合法权益。

　　谈及后续如何加快健全大型科技公司监管框架，何南野认为，“金融科技的监管主要是监管滞后问题，监管部门对科技创新及对新生事物等需要有一个接受和认识的过程，导致监管往往慢一拍”。因此，后续一方面需要监管加强对新事物、新业务的反应性，提高监管的前瞻性，同时也需要监管有更大的容忍度，包容更多的金融创新的出现，在防范风险的前提下，适度降低金融监管的力度，此外还应积极强化对头部金融控股集团的监控，防范系统性风险的发生。

　　“建议还是从制度做起，对标发达国家，我们在顶层设计上确实有提升空间，例如金融信息保护工作上，与发达国家相比我们还存在相当差距。”苏筱芮则建议，在完善顶层设计的同时，应加强行业自律、机构自治，打通外部沟通与监督渠道、倾听市场声音。

　　苏筱芮预测道，后续监管或将通过金控管理办法，对大型科技公司进一步加强

监管。11月1日，金控管理办法正式实施，现阶段还停留在准入阶段。可以预见的是，未来中长期，被纳入金控管理的公司，其事中、事后相关管理亦将完善，监管将从前、中、后各个阶段形成完善的监管体系。
"""
    summary = model.build_summarization_sample(text)
    print(summary)
    summary = model.build_text_rank_summary(text)
    print(summary)
