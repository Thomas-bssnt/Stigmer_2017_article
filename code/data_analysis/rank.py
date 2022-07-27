import numpy as np

from modules.game import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, rule, bootstrap_reps):

    map_type = "R"
    u1_def_neu, u1_neu_col = np.genfromtxt(path_data_figures + "exp/other/delimitations_players_type.txt")

    file_names = get_filenames(path_data, map_type=map_type, rule=rule)
    players = [[player for player in get_players_of_a_game(path_data, file_name)] for file_name in file_names]
    for players_game in players:
        for player in players_game:
            player.classify_player(u1_def_neu, u1_neu_col)

    mean_col, err_col, mean_neu, err_neu, mean_def, err_def = bootstrap(players, bootstrap_reps)

    np.savetxt(
        path_data_figures + f"exp/other/rule_{rule}/rk_col.txt",
        np.column_stack((mean_col, err_col.T)),
    )
    np.savetxt(
        path_data_figures + f"exp/other/rule_{rule}/rk_neu.txt",
        np.column_stack((mean_neu, err_neu.T)),
    )
    np.savetxt(
        path_data_figures + f"exp/other/rule_{rule}/rk_def.txt",
        np.column_stack((mean_def, err_def.T)),
    )


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

    return mean_col, err_col, mean_neu, err_neu, mean_def, err_def


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
