import sys

sys.path.append(
    'D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code')
import util18ji.filepath as fp
import util18ji.NMI as nmi
from util18ji.onmi import onmi
from util18ji.eq import ExtendQ
import util18ji.util as ut
from networkx.algorithms import community
import networkx as nx
import igraph as ig
from igraph import *
import math
import matplotlib.pyplot as plt
import numpy as np
import pyds
import ast
import cdlib
from cdlib import algorithms
import collections
from collections import defaultdict
import csv
import random


def createGraph(filename):
    # 读取txt、out类型文件，建立图
    G = nx.Graph()
    edges_list = []
    fp = open(filename)
    line = fp.readline()
    while line:
        edge = line.split()
        if edge[0].isdigit() and edge[1].isdigit():
            edges_list.append((int(edge[0]), int(edge[1])))
        line = fp.readline()
    fp.close()

    G.add_edges_from(edges_list)
    return G


class Community():
    ''' use set operation to optimize calculation '''

    def __init__(self, G, alpha=1.0):
        self._G = G
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0

    def add_node(self, node):
        neighbors = set(self._G.neighbors(node))
        # print("添加令居节点",neighbors , self._nodes,neighbors & self._nodes)
        node_k_in = len(neighbors & self._nodes)  # neighbor和self._nodes公有节点的数目存入node_k_in
        # print("node_k_in",node_k_in)
        node_k_out = len(neighbors) - node_k_in
        # print("node_k_out",node_k_out)
        self._nodes.add(node)
        self._k_in += 2 * node_k_in
        self._k_out = self._k_out + node_k_out - node_k_in

    def remove_node(self, node):
        neighbors = set(self._G.neighbors(node))
        community_nodes = self._nodes
        # print("community_nodes",community_nodes)
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2 * node_k_in
        self._k_out = self._k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):  # fitness适应度
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)  # vertex顶点
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha  # 幂次
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        neighbors = set(self._G.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        return new_fitness - old_fitness

    def recalculate(self):
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors

    def get_fitness(self):
        return float(self._k_in) / ((self._k_in + self._k_out) ** self._alpha)


class LFM():

    def __init__(self, G, alpha):
        self._G = G
        self._alpha = alpha

    def execute(self):
        communities = []
        node_not_include = list(self._G.nodes)
        while len(node_not_include) != 0:
            c = Community(self._G, self._alpha)
            seed = random.choice(node_not_include)
            c.add_node(seed)
            node_not_include.remove(seed)
            to_be_examined = c.get_neighbors()
            while (to_be_examined):
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)  # 计算点的适应度大于0加入，小于0删除
                    m[node] = fitness
                to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]  # 啥意思？？？
                # 适应度降序排列
                # stop condition
                if to_be_add[1] < 0.0:
                    break
                c.add_node(to_be_add[0])
                to_be_remove = c.recalculate()
                while to_be_remove != None:
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            for node in c._nodes:
                if node in node_not_include:
                    node_not_include.remove(node)
            communities.append(c._nodes)
            # print('------------------',len(node_not_include))
        return communities


if (__name__ == "__main__"):

    '''
    # 空手道数据(2) 海豚数据集(4 or 2) football数据集(12) polbooks数据集(3)
    path = [
        './NetworkWithGroundTruth/data_raw/out.ucidata-zachary',
        './NetworkWithGroundTruth/data_raw/out.dolphins',
        './NetworkWithGroundTruth/data_raw/football.txt',
        './NetworkWithGroundTruth/data_raw/polbook.txt',
        './NetworkWithGroundTruth/data_raw/riskmap.txt',
        './NetworkWithGroundTruth/data_raw/collaboration.txt'
    ]
    # path_real = [
    #     './NetworkWithGroundTruth/data_real/karate_real.txt',
    #     './NetworkWithGroundTruth/data_real/dolphins_real.txt',
    #     './NetworkWithGroundTruth/data_real/football_real.txt',
    #     './NetworkWithGroundTruth/data_real/polbooks_real.txt',
    #     './NetworkWithGroundTruth/data_real/riskmap_real.txt',
    #     './NetworkWithGroundTruth/data_real/collaboration_real.txt'
    # ]
    path_real = [
        './NetworkWithGroundTruth/data_real/karate_real.txt',
        './NetworkWithGroundTruth/data_real/dolphins_real_2.txt',
        './NetworkWithGroundTruth/data_real/football_real.txt',
        './NetworkWithGroundTruth/data_real/polbooks_real.txt',
        './NetworkWithGroundTruth/data_real/riskmap_real.txt',
        './NetworkWithGroundTruth/data_real/collaboration_real.txt'
    ]

    gid = 5
    print("邻接矩阵成图...", path[gid].split('/')[-1])
    G = createGraph(path[gid])

    #  LFM
    algorithm = LFM(G, 1.0)
    communities = algorithm.execute()

    # 获取真实社团划分
    with open(path_real[gid], "r") as text:
        line = text.readlines()
        context = str(line[0])
        context = context.replace("\n", "")
        communities_real = ast.literal_eval(context)
    real = {}
    for i in range(len(communities_real)):
        if communities_real[i] not in real:
            real[communities_real[i]] = []
        real[communities_real[i]].append(i + 1)
    real_comm = []
    for k, v in real.items():
        real_comm.append(v)

    print("社团数量：", len(communities))
    # 计算NMI
    ovnmi = onmi(communities, real_comm)
    print("ovnmi：", ovnmi)
    # 计算EQ
    # 获取边数
    edges_nums = len(nx.edges(G))
    # 获取节点度
    degree_dict = dict(nx.degree(G))
    # 获取每个节点属于的社团数
    node_coms_num = collections.defaultdict(int)
    for node_id in G.nodes():
        for comm in communities:
            if node_id in comm:
                node_coms_num[node_id] += 1

    eq = ExtendQ(G, communities, edges_nums, degree_dict, node_coms_num)
    print("eq：", eq)
    '''

    # 重叠人工网络
    # 获取网络路径
    name = "LFR4"
    # list_mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # list_om = [2, 3, 4, 5, 6]
    # list_N = [1000, 2000, 3000, 4000, 5000]
    list_on = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for on in list_on:
    # mu = 0.6
        # 获取network路径
        network_path = "D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code/NetworkWithGroundTruth/LFR_algo2/" + name + "/on=" + str(
            on) + "/network.dat"
        # 获取community路径
        community_path = "D:/Desktop/huhao/mdnotebook/__overlapping__community detection/code/NetworkWithGroundTruth/LFR_algo2/" + name + "/on=" + str(
            on) + "/community.dat"
        # 构建图
        G = nx.Graph()
        with open(network_path) as text:
            reader = csv.reader(text, delimiter="\t")
            for line in reader:
                source = int(line[0])
                target = int(line[1])
                G.add_edge(source, target)
        # 获取LFR真实社团划分（om不同）
        real_comms_dict = defaultdict(list)
        with open(community_path) as text:
            reader = csv.reader(text, delimiter="\t")
            for line in reader:
                node = int(line[0])
                labels = line[-1].split()
                for i in range(len(labels)):
                    label = int(labels[i])
                    real_comms_dict[label].append(node)
        real_comm = []
        for k, v in real_comms_dict.items():
            real_comm.append(v)

        algorithm = LFM(G, 1.0)
        communities = algorithm.execute()

        # 计算NMI
        ovnmi = onmi(communities, real_comm)
        # print("ovnmi：", ovnmi)
        # 计算EQ
        # 获取边数
        edges_nums = len(nx.edges(G))
        # 获取节点度
        degree_dict = dict(nx.degree(G))
        # 获取每个节点属于的社团数
        node_coms_num = collections.defaultdict(int)
        for node_id in G.nodes():
            for comm in communities:
                if node_id in comm:
                    node_coms_num[node_id] += 1

        eq = ExtendQ(G, communities, edges_nums, degree_dict, node_coms_num)
        # print("eq：", eq)
        # 输出onmi和eq
        print(name + " on = " + str(on) + " ovNMI = " + str(ovnmi) + " EQ = " + str(eq))
