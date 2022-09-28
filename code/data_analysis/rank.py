import numpy as np

from modules.files import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, rule, bootstrap_reps):

    u1_def_neu, u1_neu_col = np.genfromtxt(path_data_figures + "exp/classification/thresholds.txt")

    file_names = get_filenames(path_data, map_type="R", rule=rule)
    players = [[player for player in get_players_of_a_game(path_data, file_name)] for file_name in file_names]
    for players_game in players:
        for player in players_game:
            player.classify_player(u1_def_neu, u1_neu_col)

    observables = bootstrap(players, bootstrap_reps)
    for name, (mean, err) in observables.items():
        np.savetxt(
            path_data_figures + f"exp/classification/rule_{rule}/{name}.txt",
            np.column_stack((mean, err.T)),
        )

    # Statistical value
    print(f"The first player is a defector : p = {get_p_value(players, bootstrap_reps)}")


def bootstrap(players_list, bootstrap_reps):

    bs_props_col = []
    bs_props_neu = []
    bs_props_def = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_list), replace=True, size=len(players_list))

        ranks = [[] for _ in range(5)]
        for i in bs_indices:
            for player in players_list[i]:
                ranks[player.rank - 1].append(player.type)

        bs_props_col.append([rank.count("collaborator") / len(rank) for rank in ranks])
        bs_props_neu.append([rank.count("neutral") / len(rank) for rank in ranks])
        bs_props_def.append([rank.count("defector") / len(rank) for rank in ranks])

    mean_col = np.mean(bs_props_col, axis=0)
    err_col = np.abs(np.percentile(bs_props_col, [50 - 34.13, 50 + 34.13], axis=0) - mean_col)
    mean_neu = np.mean(bs_props_neu, axis=0)
    err_neu = np.abs(np.percentile(bs_props_neu, [50 - 34.13, 50 + 34.13], axis=0) - mean_neu)
    mean_def = np.mean(bs_props_def, axis=0)
    err_def = np.abs(np.percentile(bs_props_def, [50 - 34.13, 50 + 34.13], axis=0) - mean_def)

    return {
        "rk_col": (mean_col, err_col),
        "rk_neu": (mean_neu, err_neu),
        "rk_def": (mean_def, err_def),
    }


def get_p_value(players_list, bootstrap_reps):

    p_def_larger_than_p_oth = 0

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(players_list), replace=True, size=len(players_list))

        ranks_defector = []
        ranks_other = []
        for i in bs_indices:
            for player in players_list[i]:
                if player.type == "defector":
                    ranks_defector.append(player.rank)
                else:
                    ranks_other.append(player.rank)
        p_def = list(ranks_defector).count(1) / len(ranks_defector)
        p_oth = list(ranks_other).count(1) / len(ranks_other)
        if p_def > p_oth:
            p_def_larger_than_p_oth += 1

    return round(1 - p_def_larger_than_p_oth / bootstrap_reps, 4)


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
