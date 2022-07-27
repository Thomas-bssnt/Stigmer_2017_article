from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def main(path_data_figures):

    N_CLUSTER = 3

    popts_1 = np.genfromtxt(path_data_figures + "exp/other/rule_1/players_type.txt")
    popts_2 = np.genfromtxt(path_data_figures + "exp/other/rule_2/players_type.txt")
    popts = np.concatenate((popts_1, popts_2))
    u1 = popts[:, 1].reshape(-1, 1)

    ward = AgglomerativeClustering(n_clusters=N_CLUSTER, linkage="ward").fit(u1)
    limits, N_type_1, N_type_2 = get_limits_and_number(u1, ward.labels_)

    np.savetxt(path_data_figures + "exp/other/delimitations_players_type.txt", limits, fmt="%f")
    np.savetxt(path_data_figures + "exp/other/rule_1/number_players_type.txt", N_type_1, fmt="%i")
    np.savetxt(path_data_figures + "exp/other/rule_2/number_players_type.txt", N_type_2, fmt="%i")


def get_limits_and_number(u1, labels):
    N = len(u1) / 2
    numbers_1 = defaultdict(int)
    numbers_2 = defaultdict(int)
    mins = defaultdict(lambda: +100)
    maxs = defaultdict(lambda: -100)
    for i, (u, label) in enumerate(zip(u1, labels)):
        if i < N:
            numbers_1[label] += 1
        else:
            numbers_2[label] += 1
        if u < mins[label]:
            mins[label] = u
        if maxs[label] < u:
            maxs[label] = u

    types = {
        "def": min(mins, key=mins.get),
        "col": max(mins, key=mins.get),
    }
    types["neu"] = next(iter({0, 1, 2} - {types["def"], types["col"]}))

    limits = ((maxs[types["def"]] + mins[types["neu"]]) / 2, (maxs[types["neu"]] + mins[types["col"]]) / 2)
    numbers_1 = (numbers_1[types["def"]], numbers_1[types["neu"]], numbers_1[types["col"]])
    numbers_2 = (numbers_2[types["def"]], numbers_2[types["neu"]], numbers_2[types["col"]])

    return limits, numbers_1, numbers_2


if __name__ == "__main__":

    path_data_figures = "./data_figures/"

    main(path_data_figures)
