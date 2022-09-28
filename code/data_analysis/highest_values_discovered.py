import numpy as np

from modules.files import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, rule, bootstrap_reps):

    file_names = get_filenames(path_data, map_type="R", rule=rule)
    players = [[player for player in get_players_of_a_game(path_data, file_name)] for file_name in file_names]

    values_1 = []
    values_2 = []
    values_3 = []
    for players_game in players:
        values_game_1 = []
        values_game_2 = []
        values_game_3 = []
        for player in players_game:
            values_player_1 = []
            values_player_2 = []
            values_player_3 = []
            i_max_1 = -1
            v_max_1 = -1
            i_max_2 = -1
            v_max_2 = -1
            v_max_3 = -1
            for i_round in range(len(player.visits)):
                for i, v in zip(player.indices[i_round], player.visits[i_round]):
                    if v > v_max_1:
                        v_max_3 = v_max_2
                        v_max_2 = v_max_1
                        i_max_2 = i_max_1
                        v_max_1 = v
                        i_max_1 = i
                    elif v > v_max_2 and i != i_max_1:
                        v_max_3 = v_max_2
                        v_max_2 = v
                        i_max_2 = i
                    elif v > v_max_3 and i != i_max_1 and i != i_max_2:
                        v_max_3 = v
                values_player_1.append(v_max_1)
                values_player_2.append(v_max_2)
                values_player_3.append(v_max_3)
            values_game_1.append(values_player_1)
            values_game_2.append(values_player_2)
            values_game_3.append(values_player_3)
        values_1.append(np.mean(values_game_1, axis=0))
        values_2.append(np.mean(values_game_2, axis=0))
        values_3.append(np.mean(values_game_3, axis=0))

    mean_1, err_1 = bootstrap(values_1, bootstrap_reps)
    mean_2, err_2 = bootstrap(values_2, bootstrap_reps)
    mean_3, err_3 = bootstrap(values_3, bootstrap_reps)

    np.savetxt(
        path_data_figures + f"exp/observables/rule_{rule}/VB1.txt",
        np.column_stack(
            (
                np.arange(1, len(mean_1) + 1),
                mean_1,
                err_1.T,
            )
        ),
        fmt=("%d", "%f", "%f", "%f"),
    )

    np.savetxt(
        path_data_figures + f"exp/observables/rule_{rule}/VB2.txt",
        np.column_stack(
            (
                np.arange(1, len(mean_2) + 1),
                mean_2,
                err_2.T,
            )
        ),
        fmt=("%d", "%f", "%f", "%f"),
    )

    np.savetxt(
        path_data_figures + f"exp/observables/rule_{rule}/VB3.txt",
        np.column_stack(
            (
                np.arange(1, len(mean_3) + 1),
                mean_3,
                err_3.T,
            )
        ),
        fmt=("%d", "%f", "%f", "%f"),
    )


def bootstrap(observable_list, bootstrap_reps):
    bs_means = []
    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(range(len(observable_list)), replace=True, size=len(observable_list))
        bs_sample = [observable_list[i] for i in bs_indices]
        bs_means.append(np.mean(bs_sample, axis=0))
    mean = np.mean(bs_means, axis=0)
    err = np.abs(np.percentile(bs_means, [50 - 34.13, 50 + 34.13], axis=0) - mean)
    return mean, err


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
