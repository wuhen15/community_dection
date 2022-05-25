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
def run_CNM(G):
    # G = gt.get_graph(name)
    G_Zero = cG.copy_GraphToZeroBegin(G)
    origin_edges = nx.to_edgelist(G_Zero)
    new_list = []
    for edge_i in range(0, len(origin_edges)):
        node_dict = list(origin_edges)[edge_i]
        new_list.append([node_dict[0], node_dict[1]])
    g1 = Graph(new_list)  # 用变向量列表创建图形
    h1 = g1.community_fastgreedy(weights=None)
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

    if name == "santafe":
        G = ut.build_G(fp.getDataFilePathh(name))
    else:
        G = nx.read_gml(fp.getDataFilePath(name), label="id")
    #G_Zero = cG.copy_GraphToZeroBegin(G)
    res= run_CNM(G)
    #print(name + ":")
    #print(result)
    param = min(G.nodes())
    NMI_value = nmi.cal_nmi(name,res,G)
    mod = md.cal_Q(res, G)
    print(name + ":NMI = " + str(NMI_value))
    print(name + ":Q = " + str(mod))

def main_LFR(name):
    list1 =["1001"]
    for N in list1:
        list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        for MU in list:
            G = lfrtool.getNetwork(N, MU)
            G_copy = G.copy()
            res= run_CNM(G_copy)
        #print(name + ":")
        #print(result)
            param = min(G.nodes())
            NMI_value = lfrtool.LFR_nmi(N, MU, res, G)
            mod = md.cal_Q(res, G)
            print(name +"N = "+str(N)+ "MU = "+str(MU) +":NMI = " + str(NMI_value))
            print(name +"N = "+str(N)+ "MU = "+str(MU) +":Q = " + str(mod))


if __name__ == '__main__':
    # networkx = ['santafe']
    # for name in networkx:
    #     main(name)


    name = "LFRz: "
    main_LFR(name)



