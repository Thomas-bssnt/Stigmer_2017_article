import numpy as np

from modules.files import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, rule, bootstrap_reps):

    u1_def_neu, u1_neu_col = np.genfromtxt(path_data_figures + "exp/classification/thresholds.txt")
    S_max = 5420

    file_names = get_filenames(path_data, map_type="R", rule=rule)
    players = [[player for player in get_players_of_a_game(path_data, file_name)] for file_name in file_names]
    for players_game in players:
        for player in players_game:
            player.classify_player(u1_def_neu, u1_neu_col)

    #

    numbers_of_defectors = []
    team_scores = []
    for players_game in players:
        number_of_defectors = 0
        team_score = 0
        for player in players_game:
            if player.type == "defector":
                number_of_defectors += 1
            team_score += player.score / (5 * S_max)
        numbers_of_defectors.append(number_of_defectors)
        team_scores.append(team_score)

    np.savetxt(
        path_data_figures + f"exp/classification/rule_{rule}/nb_def_scores.txt",
        np.column_stack((numbers_of_defectors, team_scores)),
        fmt=("%d", "%f"),
    )

    #

    team_scores_by_number_of_defectors = [[] for _ in range(len(players[0]) + 1)]
    for number_of_defectors, team_score in zip(numbers_of_defectors, team_scores):
        team_scores_by_number_of_defectors[number_of_defectors].append(team_score)

    means = []
    medians = []
    for i in range(6):
        mean_means, err_means, mean_medians, err_medians = bootstrap(
            team_scores_by_number_of_defectors[i], bootstrap_reps
        )
        means.append([mean_means, *err_means])
        medians.append([mean_medians, *err_medians])
    np.savetxt(path_data_figures + f"exp/classification/rule_{rule}/nb_def_means.txt", means)
    np.savetxt(path_data_figures + f"exp/classification/rule_{rule}/nb_def_medians.txt", medians)

    #

    p_mean_01, p_median_01 = get_p_value(
        team_scores_by_number_of_defectors[0], team_scores_by_number_of_defectors[1], bootstrap_reps
    )
    p_mean_12, p_median_12 = get_p_value(
        team_scores_by_number_of_defectors[1], team_scores_by_number_of_defectors[2], bootstrap_reps
    )
    p_mean_02, p_median_02 = get_p_value(
        team_scores_by_number_of_defectors[0], team_scores_by_number_of_defectors[2], bootstrap_reps
    )

    print(f"0 -> 1: p_mean = {p_mean_01}, p_median = {p_median_01}")
    print(f"1 -> 2: p_mean = {p_mean_12}, p_median = {p_median_12}")
    print(f"0 -> 2: p_mean = {p_mean_02}, p_median = {p_median_02}")


def bootstrap(team_scores, bootstrap_reps):
    bs_means = []
    bs_medians = []
    for _ in range(bootstrap_reps):
        bs_team_scores = np.random.choice(team_scores, replace=True, size=len(team_scores))
        bs_means.append(np.mean(bs_team_scores))
        bs_medians.append(np.median(bs_team_scores))
    mean_means = np.median(bs_means, axis=0)
    err_means = np.abs(np.percentile(bs_means, [50 - 34.13, 50 + 34.13], axis=0) - mean_means)
    mean_medians = np.median(bs_medians, axis=0)
    err_medians = np.abs(np.percentile(bs_medians, [50 - 34.13, 50 + 34.13], axis=0) - mean_medians)
    return mean_means, err_means, mean_medians, err_medians


def get_p_value(team_scores_1, team_scores_2, bootstrap_reps):
    bs_means = []
    bs_medians = []
    for _ in range(bootstrap_reps):
        bs_team_scores_1 = np.random.choice(team_scores_1, replace=True, size=len(team_scores_1))
        bs_team_scores_2 = np.random.choice(team_scores_2, replace=True, size=len(team_scores_2))
        bs_means.append(np.mean(bs_team_scores_1) - np.mean(bs_team_scores_2))
        bs_medians.append(np.median(bs_team_scores_1) - np.median(bs_team_scores_2))
    p_mean = len([value for value in bs_means if value < 0]) / bootstrap_reps
    p_median = len([value for value in bs_medians if value < 0]) / bootstrap_reps
    return p_mean, p_median


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
