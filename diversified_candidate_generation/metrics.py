#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import data_utils.loader as LOADER

import config.const as const_util

import numpy as np
from scipy.stats import entropy
import torch
import math

class Judger(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_judger'
        self.load_features(flags_obj)
        self.metrics = flags_obj.metrics
    
    def load_features(self, flags_obj):

        loader = LOADER.CsvLoader(flags_obj)
        self.item_cate = loader.load(const_util.item_cate_feature, index_col=0)
        self.cate = self.item_cate['cid'].to_numpy()
    
    def judge(self, items, test_pos, num_test_pos, rec_items_emb):
        metric_name_map = {"hit_ratio": "Hit Ratio",
                            "recall": "Recall",
                            "precision": "Precision",
                            "ndcg": "NDCG",
                            "ilmd": "ILMD",
                            "ilad": "ILAD"}

        # results = {metric: 0.0 for metric in self.metrics}
        k = len(items[0])
        results = {}
        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            if metric in metric_name_map:
                results[metric_name_map[metric] + '@' + str(k)] = sum([f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item(), count=stat[i], items_emb=rec_items_emb[items[i]]) for i in range(len(items))])
            else:
                results[metric + '@' + str(k)] = sum([f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item(), count=stat[i], items_emb=rec_items_emb[items[i]]) for i in range(len(items))])
        return results
    
    def stat(self, items):

        stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]

        return stat


class Metrics(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_metrics'

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'ndcg': Metrics.ndcg,
            'precision': Metrics.precision,
            'coverage': Metrics.coverage,
            'entropy': Metrics.entropy,
            'gini': Metrics.gini,
            'ilmd': Metrics.ilmd,
            'ilad': Metrics.ilad
        }

        return metrics_map[metric]
    
    @staticmethod
    def ilmd(items, **kwargs):

        items_emb = torch.from_numpy(kwargs['items_emb'])

        rec_item_num = len(items) # 求得推荐的item的数量
        rec_item_emb_norm = items_emb / torch.norm(items_emb, dim=1, keepdim=True) # 归一化

        diversity_matrix = 1 - torch.mm(rec_item_emb_norm, rec_item_emb_norm.t()) # 计算多样性矩阵
        diversity_matrix[range(rec_item_num), range(rec_item_num)] = 1
        modify_matrix = diversity_matrix

        min_div = torch.min(modify_matrix).item()  
        
        return min_div
    
    @staticmethod
    def ilad(items, **kwargs):
        items_emb = torch.from_numpy(kwargs['items_emb'])

        rec_item_num = len(items) # 求得推荐的item的数量
        rec_item_emb_norm = items_emb / torch.norm(items_emb, dim=1, keepdim=True) # 归一化

        diversity_matrix = 1 - torch.mm(rec_item_emb_norm, rec_item_emb_norm.t()) # 计算多样性矩阵
        diversity_matrix[range(rec_item_num), range(rec_item_num)] = 1
        modify_matrix = diversity_matrix
        sum_i_j_div = (torch.sum(modify_matrix) - rec_item_num).item()  # 根据多样性计算公式，需要把对角线上的元素值减掉
        
        mean_div_sum = sum_i_j_div / (rec_item_num*rec_item_num - rec_item_num)
        
        return mean_div_sum
    
    @staticmethod
    def precision(items, **kwargs):
        N = len(items)
        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()
        return hit_count / N
        
    @staticmethod
    def ndcg(items, **kwargs):
        # items 是推荐tensor eg.  tensor([3226, 1099, 4983, 1040, 2229, 4443, 1555, 1221, 3240, 5452, 4119, 3193,
        #  970, 3123, 3594, 3688,  994, 2747, 1011, 2364,  112, 5084, 3096, 2211,
        # 3049, 1499, 5817, 5625, 3111, 1048,   43, 3203, 1041, 4517, 1330, 2997,
        # 1730, 1572, 4891, 1791, 1352, 5268, 4815, 1373, 3056, 4096, 4749, 5608,
        # 1250, 4974, 5060, 5276,  140, 1028, 4800, 5794, 2848, 1487, 1727,   77,
        # 3863, 1196, 5144, 1873, 4137, 5634, 5460,  374,  316, 3313, 3901, 4777,
        # 3104, 5202, 4468, 3678, 4905, 4605, 2416,  737, 3166, 5372, 5063, 5430,
        # 4541, 2135, 4190, 2945,  328, 3538, 3869, 2795, 5674,  187, 2572, 5248,
        #  633, 3766, 5041, 3396])
        test_pos = kwargs['test_pos'] # test_pos 是个list eg. [4510, 2742, 5270, 1505]
        DCG = 0
        IDCG = 0
        for n, item in enumerate(items):
            if item in test_pos:
                DCG+= 1.0/math.log(n+2)
        for n, item in enumerate(test_pos):   
            IDCG+=1.0/math.log(n+2)
        return DCG / IDCG

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count.size    # 返回类别数量

    @staticmethod
    def entropy(items, **kwargs):

        count = kwargs['count']

        return entropy(count)

    @staticmethod
    def gini(items, **kwargs):

        count = kwargs['count']
        count = np.sort(count)   # count是个list, 每个element是某测试用户推荐的item的类别个数。
        n = len(count)   # 求出有多少个类别
        cum_count = np.cumsum(count)  # 

        return (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n
