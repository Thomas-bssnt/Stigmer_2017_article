import numpy as np

from modules.gameset import GameSet


def main(path_data, path_data_folder, rule, sessions, bootstrap_reps):

    dx = 0.15
    n_bins = 25
    S_max = 5420

    games = GameSet(path_data, map_type="R", rule=rule, session=sessions)
    scores = np.array([list(game.scores_R2.values()) for game in games]) / S_max
    x_values, player_scores_dict, team_scores_dict = bootstrap(scores, bootstrap_reps, dx, n_bins)

    for filename, dict_ in (("S", player_scores_dict), ("S_team", team_scores_dict)):
        np.savetxt(
            path_data_folder + f"observables/rule_{rule}/{filename}.txt",
            np.column_stack(
                (
                    x_values,
                    dict_["pdf"][0],
                    dict_["pdf"][1].T,
                )
            ),
            fmt=("%f", "%f", "%f", "%f"),
        )

    np.savetxt(
        path_data_folder + f"observables/rule_{rule}/<S>.txt",
        [[player_scores_dict["mean"][0]] + list(player_scores_dict["mean"][1])],
        fmt=("%f", "%f", "%f"),
    )


def bootstrap(scores, bootstrap_reps, dx, n_bins):

    player_scores_pdf = []
    player_scores_means = []
    player_scores_medians = []
    team_scores_pdf = []
    team_scores_means = []
    team_scores_medians = []

    for _ in range(bootstrap_reps):
        bs_indices = np.random.choice(len(scores), replace=True, size=len(scores))
        bs_player_scores = scores[bs_indices].flatten()
        bs_team_scores = np.sum(scores[bs_indices] / len(scores[0]), axis=1)

        x_values, counts = get_hist(bs_player_scores, dx, n_bins)
        player_scores_pdf.append(counts)
        player_scores_means.append(np.mean(bs_player_scores))
        player_scores_medians.append(np.median(bs_player_scores))

        x_values, counts = get_hist(bs_team_scores, dx, n_bins)
        team_scores_pdf.append(counts)
        team_scores_means.append(np.mean(bs_team_scores))
        team_scores_medians.append(np.median(bs_team_scores))

    def _get_mean_err(list_):
        mean = np.mean(list_, axis=0)
        err = np.abs(np.percentile(list_, [50 - 34.13, 50 + 34.13], axis=0) - mean)
        return mean, err

    player_scores_dict = {
        "pdf": _get_mean_err(player_scores_pdf),
        "mean": _get_mean_err(player_scores_means),
        "median": _get_mean_err(player_scores_medians),
    }
    team_scores_dict = {
        "pdf": _get_mean_err(team_scores_pdf),
        "mean": _get_mean_err(team_scores_means),
        "median": _get_mean_err(team_scores_medians),
    }
    return x_values, player_scores_dict, team_scores_dict


def get_hist(list_, dx, n_bins):

    assert dx >= (1 - dx) / (n_bins - 1)  # to be sure that there is no gap

    x_values = np.linspace(0, 1, n_bins)

    Y = []
    for x in x_values:
        number_of_element = 0
        for element in list_:
            if x - dx / 2 < element <= x + dx / 2:
                number_of_element += 1
        Y.append(number_of_element / len(list_) / dx)

    return x_values, Y


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures + "exp/", 1, range(1, 21), bootstrap_reps)
    main(path_data, path_data_figures + "exp/", 2, range(1, 21), bootstrap_reps)

    main(path_data, path_data_figures + "exp_R1_R2/", 1, range(1, 11), bootstrap_reps)
    main(path_data, path_data_figures + "exp_R1_R2/", 2, range(1, 11), bootstrap_reps)

    main(path_data, path_data_figures + "exp_R2_R1/", 1, range(11, 21), bootstrap_reps)
    main(path_data, path_data_figures + "exp_R2_R1/", 2, range(11, 21), bootstrap_reps)
