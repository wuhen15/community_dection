from igraph import*
import matplotlib.pyplot as plt
import networkx as nx
import util.filepath as fp
import util.NMI as nmi
import util.modularity as md
import math
import util.tools as tools
import util.lfrTools as lfrtool
import util.copy_GraphToZeroBegin as cG
import util.AllNodesAddOne as AD
import util.util as ut
def run_walktrap(G):
    # G = gt.get_graph(name)
    G_Zero = cG.copy_GraphToZeroBegin(G)
    origin_edges = nx.to_edgelist(G_Zero)
    new_list = []
    for edge_i in range(0, len(origin_edges)):
        node_dict = list(origin_edges)[edge_i]
        new_list.append([node_dict[0], node_dict[1]])
    g1 = Graph(new_list)  # 用变向量列表创建图形
    h1 = g1.community_walktrap(weights=None,steps=4)
    c = list(h1.as_clustering())  # 对系统树图进行切割，按照Q值最大的标准
    # print(c)
    if min(G.nodes()) != min(G_Zero.nodes()):
        AD.AllNodesAddOne(c)
    return c
    # Q = community.modularity(G, c)
    # NMI = gt.cal_nmi(name, c, G)
    #
    # print("name: %s Q = %f, NMI = %f" % (name, Q, NMI))
def main(name):
    if name == "11":
        G = ut.build_G(fp.getDataFilePathh(name))
    else:
        G = nx.read_gml(fp.getDataFilePath(name), label="id")
    #G_Zero = cG.copy_GraphToZeroBegin(G)
    res= run_walktrap(G)
    #print(name + ":")
    #print(result)
    param = min(G.nodes())
    # NMI_value = nmi.cal_nmi(name,res,G)
    mod = md.cal_Q(res, G)
    # print(name + ":NMI = " + str(NMI_value))
    print(name + ":Q = " + str(mod))

def main_LFR(name):
    list1 =["1001"]
    for N in list1:
        list = [0.8]
        for MU in list:
            G = lfrtool.getNetwork(N, MU)
            G_copy = G.copy()
            res= run_walktrap(G_copy)
        #print(name + ":")
        #print(result)
            param = min(G.nodes())
            NMI_value = lfrtool.LFR_nmi(N, MU, res, G)
            mod = md.cal_Q(res, G)
            print(name +"N = "+str(N)+ "MU = "+str(MU) +":NMI = " + str(NMI_value))
            print(name +"N = "+str(N)+ "MU = "+str(MU) +":Q = " + str(mod))


if __name__ == '__main__':
    # networkx = ['karate','football','dolphins','polbooks','11']
    # for name in networkx:
    #     main(name)
#
#
    name = "LFRz: "
    main_LFR(name)

# import util.lfrTools as lfrtool
# from sklearn import metrics
# import igraph as ig
# from sklearn import metrics
#
# import  csv
#
# #获取igraph库可用的图，最后生成标签序列
# def getG_walktrap(N, MU):
#     path = getNetworkPath(N, MU)
#     G = ig.Graph.Read_Edgelist(path, directed=False)
#     res = list(G.community_walktrap(weights=None,steps=4).as_clustering())
#     l = len(res)
#     d = {}
#     for i in range(1, l):
#         l1 = len(res[i])
#         for j in range(0, l1):
#             d[res[i][j]] = i
#     d = sorted(d.items(), key=lambda d: d[0], reverse=False)
#     test = []
#     for i in range(0, N):
#         test.append(d[i][1])
#     return test
#
# #igraph库中的lpa
# def walktrap(n,i):
#     N = n
#     MU = i
#     ans_list = []
#
#     G = getNetwork(N, MU)
#     test = getG_walktrap(N, MU)
#     resultPath = getCommunityPath(N, MU)
#     realResult = getCommunity(resultPath, G)
#     NMI_value = metrics.normalized_mutual_info_score(test, realResult)
#     ans_list.append(NMI_value)
#     return ans_list
#
# def main_LFR(n,i):
#     temp = walktrap(n,i)
#     print("walktrap"+" u = "+str(i),temp)
# #     sum = []
# #     for i in range(0, l):
# #         sum.append(0)
# #     for i in range(0, 1):
# #         temp = walktrap(n)
# #         for j in range(0, len(temp)):
# #             sum[j] += temp[j]
# #     for i in range(0, len(sum)):
# #         sum[i] /= 1
# #         print("walktrap：", sum[i])
#
#
#
#
# def getNetwork(N, MU):
#     path = getNetworkPath(N, MU)
#     G = lfrtool.load_graph_dat(path)
#     return G
#
#
# def getCommunityPath(N,MU):
#     path = "H:/python/bishe/dataset/LFR/N=" + str(N) + "/mu=" + str(MU) + "/community.dat"
#     return path
#
# def getNetworkPath(N,MU):
#     path = "H:/python/bishe/dataset/LFR/N=" + str(N) + "/mu=" + str(MU) + "/network.dat"
#     return path
#
# def getCommunity(path, G):#获取LFR真实划分
#     real_position = [-1 for n in range(len(G.nodes()))]
#     with open(path) as text:
#         reader = csv.reader(text, delimiter="\t")
#         for line in reader:
#             source = int(line[0])
#             target = int(line[1])
#             real_position[source - 1] = target
#     return real_position
#
#
# if __name__ == '__main__':
#     # networkx = ['karate','football','dolphins','polbooks']
#     # for name in networkx:
#     #     main(name)
#     list1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
#     for i in list1:
#         name = "LFRz: "
#         main_LFR(1000,i)