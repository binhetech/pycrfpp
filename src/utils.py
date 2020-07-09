#!/usr/bin/env python
# coding=utf-8

from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


def add_tag(labels, tags=None):
    # 设置序列标注的标签体系
    # 1: BIE,S,O
    # 2: BME,S,O
    if tags is None:
        tags = ["BIE", "S", "O"]
    outs = []
    if len(labels) == 1:
        # 只有一个数
        finalTag = tags[1] + "_" + labels[-1] if labels[-1] != tags[2] else labels[-1]
        outs.append(finalTag)
    else:
        # 位置标签： B开始
        preTag = tags[0][0]
        curTag = tags[0][0]
        # 前一个类别
        preClass = labels[0]
        for i in range(1, len(labels)):
            # 当前类别
            curClass = labels[i]
            if curClass != preClass:
                if preTag == tags[0][0]:
                    preTag = tags[1]
                else:
                    preTag = tags[0][-1]
                curTag = tags[0][0]
            else:
                curTag = tags[0][1]
            # the former
            finalTag = preTag + "_" + labels[i - 1] if labels[i - 1] != tags[2] else labels[i - 1]
            outs.append(finalTag)
            # last one
            if i >= len(labels) - 1:
                if curTag == tags[0][0]:
                    curTag = tags[1]
                else:
                    curTag = tags[0][-1]
                finalTag = curTag + "_" + labels[i] if labels[i] != tags[2] else labels[i]
                outs.append(finalTag)
                break
            preTag = curTag
            preClass = labels[i]
    #     print("labels={}\nouts={}".format(labels, outs))
    return outs


def split_sent(sentIds, tokens, feats, labels):
    tokenid2sentid = {}
    for i, t in enumerate(sentIds):
        for j in range(t[0], t[1]):
            tokenid2sentid[j] = i
    newTokens, newLabels, newFeats = [[]], [[]], [[]]
    assert len(tokenid2sentid) == len(tokens)
    preId = tokenid2sentid[0]
    for idt, (t, f, lab) in enumerate(zip(tokens, feats, labels)):
        if tokenid2sentid[idt] != preId:
            newTokens.append([])
            newFeats.append([])
            newLabels.append([])
        newTokens[-1].append(t)
        newFeats[-1].append(f)
        newLabels[-1].append(lab)
        preId = tokenid2sentid[idt]
    #     print("{} newTokens={}\n{} newFeats={}\n{} newLabels={}\n".format(len(newTokens),newTokens,
    #                                                                       len(newFeats),newFeats,
    #                                                                       len(newLabels),newLabels))
    return newTokens, newFeats, newLabels


def tocrf(data, fout, hyphen="\t", labelName="LABEL"):
    outs = []
    num = 0
    for i in tqdm(data):
        # 标签转换
        labels = list(map(lambda x: str(labelName) if x == 1 else "O", i[1]["label"]))
        # 增加BIES位置标签
        labels = add_tag(labels, tags=["BIE", "S", "O"])
        #         print(i[1]["label"], "\n\n", labels)
        newTokens, newFeats, newLabels = split_sent(i[1]["sentIds"], i[1]["token"], i[1]["feat"], labels)
        for (tokenList, featList, labList) in zip(newTokens, newFeats, newLabels):
            #             print(labList)
            #             if list(set(labList))==["O"]:
            #                 continue
            for (t, f, lab) in zip(tokenList, featList, labList):
                if t[2].strip() == "":
                    continue
                content = str(t[2]) + hyphen + hyphen.join(f[1:]) + hyphen + str(lab) + "\n"
                outs.append(content)
            outs.append(" \n")
            num += 1
    print("{} sentences".format(num))
    with open(fout, "w") as f:
        f.writelines(outs)


def calc_metrics(y_true, y_pred, verbose=True):
    f1 = f1_score(y_true, y_pred, average="macro")
    print("f1={}".format(f1))
    if verbose:
        rp = classification_report(y_true, y_pred)
        print(rp)
    return f1


def run_eval(fin, head=2, verbose=True):
    with open(fin, "r", encoding="utf-8") as fo:
        data = fo.readlines()
        y_true, y_pred = [], []
        for i in data:
            i = i.split()
            if i:
                # 倒数第二列为真实值， 忽略位置信息如BIES
                y1 = i[-2][int(head):] if len(i[-2]) > 2 else i[-2]
                # 倒数第一列为预测值， 忽略位置信息如BIES
                y2 = i[-1][int(head):] if len(i[-1]) > 2 else i[-1]
                y_true.append(y1)
                y_pred.append(y2)
        f1 = calc_metrics(y_true, y_pred, verbose)
        return f1


def gen_grid_paras(parameters):
    keys, pools = [], []
    for k, v in parameters.items():
        keys.append(k)
        pools.append(tuple(v))
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    results = {}
    for idr, i in enumerate(result):
        results[idr] = {k: v for k, v in zip(keys, i)}
    print(len(results), results)
    return results
