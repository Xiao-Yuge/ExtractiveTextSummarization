# _*_coding:utf-8_*_

# @Time : 2020/11/3 16:13 
# @Author : xiaoyuge
# @File : temp.py 
# @Software: PyCharm

d = {'a':1, 'b':3, 'c':2}
print(
    dict(sorted(d.items(), key=lambda x: x[1], reverse=True)).keys()
)