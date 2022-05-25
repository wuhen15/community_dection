
from collections import defaultdict
from itertools import combinations


def f_measure(labels_real, labels):
    """
    F-measure index
    :param labels_real:
    :param labels:
    :return:
    """
    com_nodes_real = defaultdict(list)
    coms_real = set(labels_real.values())
    for com in coms_real:
        for k, v in labels_real.items():
            if v == com:
                com_nodes_real[com].append(k)
    T = []
    for nodes in com_nodes_real.values():
        T += (list(combinations(nodes, 2)))
    # print(T)
    com_nodes = defaultdict(list)
    coms = set(labels.values())
    for com in coms:
        for k, v in labels.items():
            if v == com:
                com_nodes[com].append(k)
    S = []
    for nodes in com_nodes.values():
        S += (list(combinations(nodes, 2)))
    # print(S)
    # Sort

    inter = len(list(set(T).intersection(set(S))))

    precision = 1.0 * inter / len(S)

    recall = 1.0 * inter / len(T)

    return 2.0 * precision * recall / (precision + recall)


if __name__ == "__main__":
    labels_real = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1, 16: 1, 17: 2, 18: 2, 19: 1, 20: 2, 21: 1, 22: 2, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1}
    labels = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 1, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1, 16: 1, 17: 2, 18: 2, 19: 1, 20: 2, 21: 1, 22: 2, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1}
    print(f_measure(labels_real, labels))
