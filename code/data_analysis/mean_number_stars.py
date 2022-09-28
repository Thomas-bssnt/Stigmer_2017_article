from collections import defaultdict

import numpy as np

from modules.binning import BINNING_DICT
from modules.files import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, rule, bootstrap_reps):

    u1_def_neu, u1_neu_col = np.genfromtxt(path_data_figures + "exp/classification/thresholds.txt")
    player_types = ["collaborator", "neutral", "defector", "all"]

    #

    file_names = get_filenames(path_data, map_type="R", rule=rule)

    players = [player for file_name in file_names for player in get_players_of_a_game(path_data, file_name)]
    for player in players:
        player.classify_player(u1_def_neu, u1_neu_col)

    mean, err = bootstrap(players, player_types, bootstrap_reps)

    np.savetxt(
        path_data_figures + f"exp/classification/rule_{rule}/MSN_all.txt",
        np.column_stack(
            (
                list(mean["all"].keys()),
                list(mean["all"].values()),
                *np.array(list(err["all"].values())).T,
            )
        ),
        fmt=("%f", "%f", "%f", "%f"),
    )

    np.savetxt(
        path_data_figures + f"exp/classification/rule_{rule}/MSN_col.txt",
        np.column_stack(
            (
                list(mean["collaborator"].keys()),
                list(mean["collaborator"].values()),
                *np.array(list(err["collaborator"].values())).T,
            )
        ),
        fmt=("%f", "%f", "%f", "%f"),
    )

    np.savetxt(
        path_data_figures + f"exp/classification/rule_{rule}/MSN_neu.txt",
        np.column_stack(
            (
                list(mean["neutral"].keys()),
                list(mean["neutral"].values()),
                *np.array(list(err["neutral"].values())).T,
            )
        ),
        fmt=("%f", "%f", "%f", "%f"),
    )

    np.savetxt(
        path_data_figures + f"exp/classification/rule_{rule}/MSN_def.txt",
        np.column_stack(
            (
                list(mean["defector"].keys()),
                list(mean["defector"].values()),
                *np.array(list(err["defector"].values())).T,
            )
        ),
        fmt=("%f", "%f", "%f", "%f"),
    )


def binning(dict_, binning_dict):
    new_dict = defaultdict(list)
    for v, new_v in binning_dict.items():
        new_dict[new_v] += dict_[v]
    return new_dict


def bootstrap(players, player_types, bootstrap_reps):

    bs_means = {player_type: defaultdict(list) for player_type in player_types}

    for _ in range(bootstrap_reps):

        bs_players = np.random.choice(players, replace=True, size=len(players))

        for player_type in player_types:
            numbers_of_stars_played = defaultdict(list)
            for player in bs_players:
                if player.type == player_type or player_type == "all":
                    for value, stars in player.numbers_of_stars_played.items():
                        numbers_of_stars_played[value] += stars
            numbers_of_stars_played_binned = binning(numbers_of_stars_played, BINNING_DICT)
            for value, stars in numbers_of_stars_played_binned.items():
                bs_means[player_type][value].append(np.mean(stars))

    mean_1 = {
        player_type: {value: np.nanmean(bs_mean) for value, bs_mean in bs_means[player_type].items()}
        for player_type in player_types
    }
    err_1 = {
        player_type: {
            value: np.abs(np.nanpercentile(bs_mean, [50 - 34.13, 50 + 34.13]) - mean_1[player_type][value])
            for value, bs_mean in bs_means[player_type].items()
        }
        for player_type in player_types
    }

    return mean_1, err_1


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 1000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
