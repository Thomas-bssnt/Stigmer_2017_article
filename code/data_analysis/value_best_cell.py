from collections import defaultdict
from json import load

import numpy as np

from modules.files import get_filenames


def main(path_data, path_data_figures, rule, bootstrap_reps):

    map_type = "R"

    value_cells_sorted = []

    file_names = get_filenames(path_data, map_type=map_type, rule=rule)
    for file_name in file_names:

        path_in_file = path_data + f"/session_{file_name[1:3]}/in/{file_name}.json"
        with open(path_in_file) as in_file:
            in_data = load(in_file)

        path_out_file = path_data + f"/session_{file_name[1:3]}/out/{file_name}.csv"
        out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)

        vCellsPlayed = defaultdict(lambda: [[] for _ in range(in_data["numberRounds"])])
        for round_, playerId, _, _, value, _, _ in out_data:
            vCellsPlayed[playerId][round_ - 1].append(value)

        vCellsSortedByValue = np.zeros((in_data["numberCasesPerRound"], in_data["numberRounds"]))
        for playerId in vCellsPlayed.keys():
            for round_ in range(len(vCellsPlayed[playerId])):
                vCellsSortedByValue[:, round_] += sorted(vCellsPlayed[playerId][round_])
        vCellsSortedByValue /= in_data["numberPlayers"]

        value_cells_sorted.append(vCellsSortedByValue)

    value_cells_sorted = np.array(value_cells_sorted)

    mean, err = bootstrap(value_cells_sorted, bootstrap_reps)
    for i in range(3):

        np.savetxt(
            path_data_figures + f"exp/other/rule_{rule}/V{3-i}.txt",
            np.column_stack(
                (
                    np.arange(1, len(mean[i]) + 1),
                    mean[i],
                    err[:, i, :].T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )


def bootstrap(observable_list, bootstrap_reps):
    bs_values = []
    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(range(len(observable_list)), replace=True, size=len(observable_list))
        bs_sample = observable_list[bs_indices]
        bs_values.append(np.mean(bs_sample, axis=0))
    mean = np.mean(bs_values, axis=0)
    err = np.abs(np.percentile(bs_values, [50 - 34.13, 50 + 34.13], axis=0) - mean)
    return mean, err


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 1000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
