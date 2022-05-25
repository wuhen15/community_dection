import pandas as pd


def print_res(res, name):
    file_path = r'D:\Desktop\huhao\mdnotebook\__overlapping__community detection\notebook\毕业算法1\real_net_res_graph\net_structure'
    label_ls = []
    weg_ls = []
    com_ls = []
    for k in range(len(res)):
        for i in range(len(res[k])):
            label_ls.append(res[k][i] - 1)
            weg_ls.append(1)
            com_ls.append(k)
    df_can = pd.DataFrame()
    df_can['Label'] = label_ls
    df_can['Weight'] = weg_ls
    df_can['Com'] = com_ls
    df_can.to_csv(file_path + "\\" + name + "社团结果.csv")


if __name__ == "__main__":
    name = 'Riskmap'
    res = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 29], [10, 11, 13, 14],
           [12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28],
           [15, 16, 26, 33, 34, 35, 36, 37, 38], [30, 31, 32],
           [39, 40, 41, 42]]
    print_res(res, name)
