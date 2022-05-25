import networkx as nx
from networkx import jaccard_coefficient
import relate_code.util.filepath as fp
import relate_code.util.NMI as nmi
import relate_code.util.modularity as md
# import math
# import util.tools as tools
import relate_code.util.lfrTools as lfrtool
import numpy as np
import relate_code.util as ut
from time import *
#得到节点度大小排序序列字典
def get_hexinzz(G):
    list_du = {}
    for i in G.nodes():
        list_du[i]=G.degree[i]
    list_du = sorted(list_du.items(), key=lambda x: x[1],reverse=True)
    return list_du
def get_jaccard_coefficients(G,u,v):
    if u == v:
        return 0
    union_size = len(set(G[u]) | set(G[v]))
    #if union_size == 0:
        #return 0
    return (len(list(nx.common_neighbors(G, u, v)))) / (union_size)

#建立邻接矩阵
def similarityMatrix(G):
    N =len(G)
    simMatrix = np.zeros((N,N))

    if min(G.nodes())==1:
        for i in range(min(G.nodes()),max(G.nodes())+1):

            for j in range(min(G.nodes()),max(G.nodes())+1):
                    if j in G[i]:
                        simMatrix[i-1][j-1] = 1
    else:
        for i in range(min(G.nodes()), max(G.nodes()) + 1):

            for j in range(min(G.nodes()), max(G.nodes()) + 1):
                    if j in G[i]:
                        simMatrix[i][j] = 1

    return simMatrix
#求任意两个社团相似度
def get_xiangsi(G,list1,list2):
    sim =similarityMatrix(G)
    m = G.size()
    sum = 0
    xiangsi = 0
    if min(G.nodes())==1:
        for i in list1:
            for j in list2:
                # if j in G[i]:
                    sum = sum+(sim[i-1][j-1]-G.degree[i]*G.degree[j]/(2*m))
        xiangsi = sum/(2*m)
    else:
        for i in list1:
            for j in list2:
                # if j in G[i]:
                    sum = sum+(sim[i][j]-G.degree[i]*G.degree[j]/(2*m))
        xiangsi = sum / (2 * m)
    return xiangsi
# G =nx.karate_club_graph()
# print(get_hexinzz(G))
def xianglin(G,list3,list4):
    for i in list3:
        for j in list4:
            if i in G[j]:
                return 1
    return 0

def community(G):
    #RIA算法
    G_1 = G.copy()
    label = []
    if min(G.nodes())==1:
        label = [0 for x in range(0,len(G)+1)]
    else:
        label = [0 for x in range(0,len(G))]
    list_chushi = get_hexinzz(G)
    list_jiedian =[]
    for tup in list_chushi:
        list_jiedian.append(tup[0])
    RC = []

    if min(G.nodes()) == 1:
        for i in list_jiedian:

            if label[i] == 0:
                C =[]
                C.append(i)

                label[i] = 1
                for j in G[i]:
                    if label[j] == 0:
                        label[j] = 1
                        C.append(j)
                RC.append(C)
    else:
        for i in list_jiedian:
            if label[i] == 0:
                C = []
                C.append(i)
                label[i] =1
                for j in G[i]:
                    if label[j] == 0:
                        label[j] = 1
                        C.append(j)

                RC.append(C)

    #边界重建
    # print("RC",RC)
    tmp = {}
    n = 0
    for li in RC:
        n = n + 1
        for num in li:
            tmp[num] = n
    # print ("tmp",tmp)
    # 求社团划分后所有的边界节点
    label1 = [0 for x in range(0,len(G)+1)]#建立节点标签v.inter
    bianjielist = []
    for i in G.nodes():
        for j in G[i]:
            if tmp[i] != tmp[j]:
                bianjielist.append(i)
                # label[i+1]=1

    # bj = []
    # print("tmp",tmp)
    bianjielist = set(bianjielist)
    bianjielist = list(bianjielist)
    # print("bianjielist",bianjielist)
    bj = []
    res = {}
    while len(bianjielist)!= len(bj):
        dic_i = {}
        Lnode =[]#代表

        for i in bianjielist:
            dic_neighbor = {}
            if label1[i] == 0:
                label1[i]=1
                bj.append(i)
                for j in G[i]:
                    Lnode.append(j)

                    degree = 0.5* G.degree(j) / (len(G)-1)  # j的度
                    sim = 0.5 * get_jaccard_coefficients(G, i, j)
                    if tmp[j] != tmp[i]:

                        if (tmp[j] not in dic_neighbor.keys()):
                            dic_neighbor[tmp[j]] = degree + sim
                        else:
                            dic_neighbor[tmp[j]] = dic_neighbor[tmp[j]] + degree + sim
                    if tmp[j] == tmp[i]:
                        # j的度
                        if (tmp[j] not in dic_neighbor.keys()):
                            dic_neighbor[tmp[j]] = degree + sim
                        else:
                            dic_neighbor[tmp[j]] = dic_neighbor[tmp[j]] + degree + sim
                # for key, value in dic_neighbor.items():
                    num = max(zip(dic_neighbor.values(),dic_neighbor.keys()))
                    if  tmp[i]!= num[1]:
                        tmp[i] = num[1]
                    #if value_1 == num and key_1 != tmp[key]:
                    #             tmp[key] = key_1
                # dic_i[i] = dic_neighbor#dic_i[i]是{ i:{   }       }的形式，代表边界节点i在各社区的隶属度
                Lnode = set(Lnode)
                Lnode = list(Lnode)
                # print("dic_i",dic_i)
                # print("Lnode",Lnode)
        # for key, value in dic_i.items():
        #     num = max(value.values())
        #     for key_1, value_1 in value.items():
        #         if value_1 == num and key_1 != tmp[key]:
        #             tmp[key] = key_1


        for a in Lnode:
            if label1[a] == 0:
                for num in G[a]:
                    if tmp[a]!=tmp[num]:
                        bianjielist.append(a)
                        break
    # print("tmphou", tmp)
    for node, communityNum in tmp.items():
        if communityNum in res.keys():
            res[communityNum].append(node)
        else:
            res[communityNum] = [node]
    # print("res",res)
    result = []
    for com in res.values():
        result.append(com)
    # print("result",result)
    smallcommunity = []
    bigcommunity = []
    # print("合并开始---------------------")
    # 取出社团节点数目小于3的所有社团，将其存放在一个列表中
    for i in range(len(result)):
        if len(result[i]) <= 2:
            smallcommunity.append(result[i])
        else:
            bigcommunity.append(result[i])
    hebin = {}
    for i in range(len(smallcommunity)):
        edge_num = []
        for j in range(len(bigcommunity)):
            if xianglin(G,smallcommunity[i],bigcommunity[j])==1:
                edge_num.append(get_xiangsi(G, smallcommunity[i], bigcommunity[j]))
        tmp = 0
        index = 0
        for a in range(len(edge_num)):
            if (tmp < edge_num[a]):
                tmp = edge_num[a]
                index = a
        hebin[i] = index
    for k, v in hebin.items():
        bigcommunity[v].extend(smallcommunity[k])
    # print(bigcommunity)
    return bigcommunity


def main(name):
    if name == "facebook":
        G = ut.build_G(fp.getDataFilePathh(name))
    else:
        G = nx.read_gml(fp.getDataFilePath(name), label="id")
    G1 = G.copy()
    res = community(G1)
    # print(name + ":")
    # print(result)
    param = min(G.nodes())
    NMI_value = nmi.cal_nmi(name, res, G)
    mod = md.cal_Q(res, G)
    print(name + ":NMI = " + str(NMI_value))
    print(name + ":Q = " + str(mod))


def main_LFR(name):
    N = 1001
    list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for MU in list:
        G = lfrtool.getNetwork(N, MU)
        G_copy = G.copy()
        res = community(G_copy)
        # print(name + ":")
        # print(result)
        param = min(G.nodes())
        NMI_value = lfrtool.LFR_nmi(N, MU, res, G)
        mod = md.cal_Q(res, G)
        print(name + "MU = " + str(MU) + ":NMI = " + str(NMI_value))
        print(name + "MU = " + str(MU) + ":Q = " + str(mod))


if __name__ == '__main__':
    # begin_time = time()
    # networkx = ['karate','football','dolphins','polbooks']
    # for name in networkx:
    #     main(name)
    # end_time = time()
    # run_time = end_time-begin_time
    # print("yunxin",run_time)
    #
    name = "LFR: N = 5000"
    main_LFR(name)
    # list1 = [2,1,3]
    # list2 = [1,2,3]
    # print(list1 == list2)
