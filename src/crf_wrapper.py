#!/usr/bin/env python
# coding=utf-8
"""
本模块用于利用CRF++模型进行序列标注的python封装.

本模块中包含以下类:
    CrfWrapper: 加载CRF++模型进行序列标注的python封装类.

本模块中包含以下方法:
    None

本模块中包含以下属性:
    None

"""

import pandas as pd
import CRFPP


class CrfWrapper(object):
    """
    利用CRF++模型进行序列标注的python封装类.

    本类中包含以下方法:
        __init__: 初始化方法
        predict: 进行序列标注标签预测
        predict_proba: 进行序列标注标签概率预测

    本类中包含以下属性:
        None

    """

    def __init__(self, pathModel=r"./crfpp_model.bin"):
        """
        初始化方法.

        Args:
            pathModel: string, CRF++模型路径

        Return:
            None

        """
        # -v 3: access deep information like alpha, beta, prob
        # -n N: enable N best output. N should be >= 2
        self.tagger = CRFPP.Tagger("-m " + pathModel + " -v 3 -n 2")
        # clear internal context
        self.tagger.clear()

        # 输入特征列数
        print("column size: {}".format(self.tagger.xsize()))
        # 输入tokens(即句子)数
        # print("token size: {}".format(self.tagger.size()))
        # 序列标注输出类别数
        print("tag size: {}".format(self.tagger.ysize()))
        self.tagsList = [self.tagger.yname(i) for i in range(self.tagger.ysize())]
        print("tag set: {}".format(self.tagsList))
        self.classes_ = self.tagsList
        return

    def predict(self, tokens):
        """
        序列标注标签预测.

        Args:
            tokens: list of feats, 输入tokens的特征数据列表

        return:
            labels: list of string, 输出对应tokens的序列标注标签列表

        """
        # clear internal context
        self.tagger.clear()
        # 输入tokens
        # print("input size: {}, tokens={}".format(len(tokens), tokens))
        # add context
        for i in tokens:
            self.tagger.add(i)
        size = self.tagger.size()
        # ysize = self.tagger.ysize()
        # print("size={}, ysize={}".format(size, ysize))
        # parse and change internal stated as 'parsed'
        self.tagger.parse()
        # print("conditional prob={}, log(Z)=={}".format(self.tagger.prob(), self.tagger.Z()))
        labels = [self.tagger.y2(i) for i in range(0, size)]
        return labels

    def predict_proba(self, tokens):
        """
        序列标注标签概率预测.

        Args:
            tokens: list of feats, 输入tokens的特征数据列表

        return:
            probs: list of array(dim=size of self.tagsList), 输出对应tokens的序列标注标签概率列表

        """
        # clear internal context
        self.tagger.clear()
        # add context
        for i in tokens:
            self.tagger.add(i)
        size = self.tagger.size()
        ysize = self.tagger.ysize()
        # 标注器解析
        self.tagger.parse()
        probs = []
        for i in range(0, size):
            prob = [self.tagger.prob(i, j) for j in range(0, ysize)]
            p = pd.Series(prob, index=self.tagsList)
            p = p.reindex(sorted(self.tagsList))
            # 输出对应token的排序后的tag类别概率，与分类模型保持一致性
            probs.append(p.values)
        return probs
