from collections import defaultdict
from json import load

import numpy as np

from modules.files import get_filenames


def main(path_data, path_data_figures, rule, bootstrap_reps):

    playBestCell = []
    playSecondBestCell = []
    playThirdBestCell = []
    explore = []

    file_names = get_filenames(path_data, map_type="R", rule=rule)
    for file_name in file_names:

        path_in_file = path_data + f"/session_{file_name[1:3]}/in/{file_name}.json"
        with open(path_in_file) as in_file:
            in_data = load(in_file)

        path_out_file = path_data + f"/session_{file_name[1:3]}/out/{file_name}.csv"
        out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)

        stars_played = np.zeros((in_data["numberRounds"], in_data["mapSize"] ** 2), dtype=int)
        iCellsPlayed = defaultdict(lambda: [[] for _ in range(in_data["numberRounds"])])
        vCellsPlayed = defaultdict(lambda: [[] for _ in range(in_data["numberRounds"])])
        for round_, playerId, mapX, mapY, value, numberStars, _ in out_data:
            stars_played[round_ - 1, mapY * in_data["mapSize"] + mapX] += numberStars
            iCellsPlayed[playerId][round_ - 1].append(mapY * in_data["mapSize"] + mapX)
            vCellsPlayed[playerId][round_ - 1].append(value)

        iCellsSortedByValue = {playerId: [] for playerId in iCellsPlayed.keys()}
        vCellsSortedByValue = {playerId: [] for playerId in iCellsPlayed.keys()}
        for playerId in iCellsPlayed.keys():
            for round_ in range(len(iCellsPlayed[playerId])):
                iSort = np.argsort(vCellsPlayed[playerId][round_])
                iCellsSortedByValue[playerId].append([iCellsPlayed[playerId][round_][i] for i in iSort])
                vCellsSortedByValue[playerId].append([vCellsPlayed[playerId][round_][i] for i in iSort])

        playBestCellPlayer = np.zeros(in_data["numberRounds"])
        playSecondBestCellPlayer = np.zeros(in_data["numberRounds"])
        playThirdBestCellPlayer = np.zeros(in_data["numberRounds"])
        explorePlayer = np.zeros(in_data["numberRounds"])
        for playerId in iCellsPlayed.keys():
            explorePlayer[0] += in_data["numberCasesPerRound"]
            for round_ in range(1, in_data["numberRounds"]):
                for iTurn in range(in_data["numberCasesPerRound"]):
                    iCell = iCellsPlayed[playerId][round_][iTurn]
                    if iCell == iCellsSortedByValue[playerId][round_ - 1][-1]:
                        playBestCellPlayer[round_] += 1
                    elif iCell == iCellsSortedByValue[playerId][round_ - 1][-2]:
                        playSecondBestCellPlayer[round_] += 1
                    elif iCell == iCellsSortedByValue[playerId][round_ - 1][-3]:
                        playThirdBestCellPlayer[round_] += 1
                    else:
                        explorePlayer[round_] += 1
        playBestCell.append(playBestCellPlayer / in_data["numberPlayers"])
        playSecondBestCell.append(playSecondBestCellPlayer / in_data["numberPlayers"])
        playThirdBestCell.append(playThirdBestCellPlayer / in_data["numberPlayers"])
        explore.append(explorePlayer / in_data["numberPlayers"])

    playBestCell = np.array(playBestCell)
    playSecondBestCell = np.array(playSecondBestCell)
    playThirdBestCell = np.array(playThirdBestCell)
    explore = np.array(explore)

    for i, list_ in enumerate([playBestCell, playSecondBestCell, playThirdBestCell], 1):
        mean, err = bootstrap(list_, bootstrap_reps)

        np.savetxt(
            path_data_figures + f"exp/observables/rule_{rule}/B{i}.txt",
            np.column_stack(
                (
                    np.arange(2, len(mean) + 1),
                    mean[1:],
                    err[:, 1:].T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )


def bootstrap(observable_list, bootstrap_reps):
    bs_probas = []
    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(range(len(observable_list)), replace=True, size=len(observable_list))
        bs_sample = observable_list[bs_indices]
        bs_probas.append(np.mean(bs_sample, axis=0))
    mean = np.mean(bs_probas, axis=0)
    err = np.abs(np.percentile(bs_probas, [50 - 34.13, 50 + 34.13], axis=0) - mean)
    return mean, err


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
