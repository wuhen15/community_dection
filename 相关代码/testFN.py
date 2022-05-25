import networkx as nx
import matplotlib.pyplot as plt
from networkx import jaccard_coefficient
import util.filepath as fp
import util.NMI as nmi
import util.modularity as md
import math
import util.tools as tools
import util.lfrTools as lfrtool
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
def Run_FN(G):
    G_1 = G.copy()
    result_FN = community.greedy_modularity_communities(G_1)
    result = []
    for c in (result_FN):
        result.append(list(c))
    return result
def main(name):
    G = nx.read_gml(fp.getDataFilePath(name), label="id")
    res = Run_FN(G)
    # print(name + ":")
    # print(result)
    param = min(G.nodes())
    NMI_value = nmi.cal_nmi(name, res, G)
    mod = md.cal_Q(res, G)
    print(name + ":NMI = " + str(NMI_value))
    print(name + ":Q = " + str(mod))


def main_LFR(name):
    list1 =["5000"]
    for N in list1:
        list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        for MU in list:
            G = lfrtool.getNetwork(N, MU)
            G_copy = G.copy()
            res = Run_FN(G_copy)
            # print(name + ":")
            # print(result)
            param = min(G.nodes())
            NMI_value = lfrtool.LFR_nmi(N, MU, res, G)
            mod = md.cal_Q(res, G)
            print(name + "N = " + str(N) + "MU = " + str(MU) + ":NMI = " + str(NMI_value))
            print(name + "N = " + str(N) + "MU = " + str(MU) + ":Q = " + str(mod))


if __name__ == '__main__':
    # networkx = ['karate','football','dolphins','polbooks']
    # for name in networkx:
    #     main(name)

    name = "LFRz: "
    main_LFR(name)
