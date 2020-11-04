# _*_coding:utf-8_*_

# @Time : 2020/11/3 16:13 
# @Author : xiaoyuge
# @File : page_rank.py
# @Software: PyCharm

import numpy as np

DAMPING_FACTOR = 0.15


def calc_page_rank(lst, b):
    count = {}
    ele2ind = {}
    ele_adj = {}
    for item in lst:
        if item[1] not in count:
            count[item[1]] = 0
            ele2ind[item[1]] = len(ele2ind)
        if item[0] not in count:
            count[item[0]] = 1
            ele_adj[item[0]] = [item[1]]
            ele2ind[item[0]] = len(ele2ind)
        elif count[item[0]] == 0:
            count[item[0]] = 1
            ele_adj[item[0]] = [item[1]]
        else:
            count[item[0]] += 1
            ele_adj[item[0]].append(item[1])
    A = np.ones((len(count), len(count))) / len(count)
    B = A.copy()
    for ele in ele_adj:
        A[:, ele2ind[ele]] = 0
        for adj in ele_adj[ele]:
            A[ele2ind[adj]][ele2ind[ele]] = 1 / len(ele_adj[ele])

    damping_factor = 0.15
    M = (1-damping_factor)*A + damping_factor*B
    it = 0
    v = np.array([1/M.shape[0]]*M.shape[0])
    min_bias = np.array([b]*v.shape[0], dtype=np.float32)
    while True:
        it += 1
        temp = np.matmul(M, v)
        if (temp - v < min_bias).all():
            break
        v = temp
    page_rank = dict(sorted(zip(ele2ind.keys(), v.tolist()), key=lambda x: x[1], reverse=True))
    return it, page_rank


if __name__ == "__main__":
    import random
    min_bias = 1e-10
    # it, page_rank = calc_page_rank([
    #     ('A', 'B'),
    #     ('A', 'C'),
    #     ('B', 'D'),
    #     ('C', 'A'),
    #     ('C', 'B'),
    #     ('C', 'D'),
    #     ('D', 'C')
    # ], min_bias)

    # it, page_rank = calc_page_rank([
    #     ('A', 'C'),
    #     ('B', 'C')
    # ], min_bias)

    random_graph = set([(random.randint(0, 500), random.randint(0, 500)) for i in range(300000)])
    it, page_rank = calc_page_rank(random_graph, min_bias)

    print('{}, {}, {}'.format(it, page_rank, sum(page_rank.values())))
