from igraph import*
import matplotlib.pyplot as plt
import networkx as nx
import util.filepath as fp
import util.NMI as nmi
import util.modularity as md
import math
import util.tools as tools
import util.lfrTools as lfrtool
from sklearn import metrics
import igraph as ig
from sklearn import metrics
import util.copy_GraphToZeroBegin as cG
import util.AllNodesAddOne as AD
import  csv
# def run_LPA(G):
#     # G = gt.get_graph(name)
#     G_Zero = cG.copy_GraphToZeroBegin(G)
#     origin_edges = nx.to_edgelist(G_Zero)
#     new_list = []
#     for edge_i in range(0, len(origin_edges)):
#         node_dict = list(origin_edges)[edge_i]
#         new_list.append([node_dict[0], node_dict[1]])
#     g1 = Graph(new_list)  # 用变向量列表创建图形
#     h1 = g1.community_label_propagation()
#     c = list(h1)  # 对系统树图进行切割，按照Q值最大的标准
#     # print(c)
#     if min(G.nodes()) != min(G_Zero.nodes()):
#         AD.AllNodesAddOne(c)
#     return c
#     # Q = community.modularity(G, c)
#     # NMI = gt.cal_nmi(name, c, G)
#     #
#     # print("name: %s Q = %f, NMI = %f" % (name, Q, NMI))
# def main(name):
#     count = 150
#     sum_Q= 0.0
#     sum_NMI = 0.0
#     for index in range(count):
#
#         G = nx.read_gml(fp.getDataFilePath(name), label="id")
#         #G_Zero = cG.copy_GraphToZeroBegin(G)
#         res= run_LPA(G)
#         #print(name + ":")
#         #print(result)
#         param = min(G.nodes())
#         sum_NMI += nmi.cal_nmi(name,res,G)
#         sum_Q += md.cal_Q(res, G)
#
#     Q = sum_Q / count
#     NMI = sum_NMI / count
#
#     print(name + ":NMI = " + str(NMI))
#     print(name + ":Q = " + str(Q))
#
# def main_LFR(name):
#     list1 =[1000]
#     for N in list1:
#         list = [0.1,0.2,0.3]
#         for MU in list:
#
#             G = lfrtool.getNetwork(N, MU)
#             count = 5
#             sum_Q = 0.0
#             sum_NMI = 0.0
#             for index in range(count):
#                 G_copy = G.copy()
#                 res= run_LPA(G_copy)
#             #print(name + ":")
#             #print(result)
#                 param = min(G.nodes())
#                 sum_NMI += lfrtool.LFR_nmi(N, MU, res, G)
#                 sum_Q += md.cal_Q(res, G)
#             NMI = sum_NMI/count
#             Q = sum_Q/count
#             print(name +"N = "+str(N)+ "MU = "+str(MU) +":NMI = " + str(NMI))
#             print(name +"N = "+str(N)+ "MU = "+str(MU) +":Q = " + str(Q))
#

#获取igraph库可用的图，最后生成标签序列
def getG_lpa(N, MU):
    path = getNetworkPath(N, MU)
    G = ig.Graph.Read_Edgelist(path, directed=False)
    res = G.community_label_propagation()
    l = len(res)
    d = {}
    for i in range(1, l):
        l1 = len(res[i])
        for j in range(0, l1):
            d[res[i][j]] = i
    d = sorted(d.items(), key=lambda d: d[0], reverse=False)
    test = []
    for i in range(0, N):
        test.append(d[i][1])
    return test

#igraph库中的lpa
def lpa(n):
    N = n
    list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    ans_list = []
    for MU in list:
        G = getNetwork(N, MU)
        test = getG_lpa(N, MU)
        resultPath = getCommunityPath(N, MU)
        realResult = getCommunity(resultPath, G)
        NMI_value = metrics.normalized_mutual_info_score(test, realResult)
        ans_list.append(NMI_value)
    return ans_list

def main_LFR(n, l):
    sum = []
    for i in range(0, l):
        sum.append(0)
    for i in range(0, 10):
        temp = lpa(n)
        for j in range(0, len(temp)):
            sum[j] += temp[j]
    for i in range(0, len(sum)):
        sum[i] /= 10
        print("lpa平均100次：", sum[i])




def getNetwork(N, MU):
    path = getNetworkPath(N, MU)
    G = lfrtool.load_graph_dat(path)
    return G


def getCommunityPath(N,MU):
    path = "D:/bishe/dataset/LFR/N=" + str(N) + "/mu=" + str(MU) + "/community.dat"
    return path

def getNetworkPath(N,MU):
    path = "D:/bishe/dataset/LFR/N=" + str(N) + "/mu=" + str(MU) + "/network.dat"
    return path

def getCommunity(path, G):#获取LFR真实划分
    real_position = [-1 for n in range(len(G.nodes()))]
    with open(path) as text:
        reader = csv.reader(text, delimiter="\t")
        for line in reader:
            source = int(line[0])
            target = int(line[1])
            real_position[source - 1] = target
    return real_position


if __name__ == '__main__':
#     # networkx = ['karate','football','dolphins','polbooks']
#     # for name in networkx:
#     #     main(name)
#
#
    name = "LFRz: "
    main_LFR(1000,10)