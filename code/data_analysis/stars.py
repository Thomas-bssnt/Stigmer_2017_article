from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

from modules.binning import BINNING_DICT
from modules.files import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, bootstrap_reps):

    u1_def_neu, u1_neu_col = np.genfromtxt(path_data_figures + "exp/classification/thresholds.txt")

    #############################################################################

    file_names = get_filenames(path_data, map_type="R", rule=[1, 2])

    players = [player for file_name in file_names for player in get_players_of_a_game(path_data, file_name)]
    for player in players:
        player.classify_player(u1_def_neu, u1_neu_col)

    values = np.unique(list(BINNING_DICT.values()))
    stars = np.arange(6)

    probas_col, popt_col = bootstrap(values, stars, players, "collaborator", bootstrap_reps)
    probas_neu, popt_neu = bootstrap(values, stars, players, "neutral", bootstrap_reps)
    probas_def, popt_def = bootstrap(values, stars, players, "defector", bootstrap_reps)
    print(f"{popt_col = }")
    print(f"{popt_neu = }")
    print(f"{popt_def = }")

    values_model = np.arange(max(values) + 1)

    probas_model_col = get_probas_model(values_model, popt_col["mean"], "tanh")
    probas_model_neu = get_probas_model(values_model, popt_neu["mean"], "linear")
    probas_model_def = get_probas_model(values_model, popt_def["mean"], "tanh")

    #############################################################################

    np.savetxt(
        path_data_figures + "model/parameters/stars_col.txt",
        popt_col["mean"][None],
        fmt="%f",
    )
    np.savetxt(
        path_data_figures + "model/parameters/stars_neu.txt",
        popt_neu["mean"][None],
        fmt="%f",
    )
    np.savetxt(
        path_data_figures + "model/parameters/stars_def.txt",
        popt_def["mean"][None],
        fmt="%f",
    )

    #############################################################################

    for i, P in ((0, "P0"), (1, "P1234"), (5, "P5")):
        np.savetxt(
            path_data_figures + f"exp/classification/P{P}_col.txt",
            np.column_stack(
                (
                    values,
                    probas_col["mean"][:, i],
                    probas_col["err"][:, :, i].T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )
        np.savetxt(
            path_data_figures + f"exp/classification/P{P}_neu.txt",
            np.column_stack(
                (
                    values,
                    probas_neu["mean"][:, i],
                    probas_neu["err"][:, :, i].T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )
        np.savetxt(
            path_data_figures + f"exp/classification/P{P}_def.txt",
            np.column_stack(
                (
                    values,
                    probas_def["mean"][:, i],
                    probas_def["err"][:, :, i].T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )

        np.savetxt(
            path_data_figures + f"model/classification/P{P}_col.txt",
            np.column_stack((values_model, probas_model_col[:, i])),
            fmt=("%d", "%f"),
        )
        np.savetxt(
            path_data_figures + f"model/classification/P{P}_neu.txt",
            np.column_stack((values_model, probas_model_neu[:, i])),
            fmt=("%d", "%f"),
        )
        np.savetxt(
            path_data_figures + f"model/classification/P{P}_def.txt",
            np.column_stack((values_model, probas_model_def[:, i])),
            fmt=("%d", "%f"),
        )

    #############################################################################

    for rule in [1, 2]:

        file_names = get_filenames(path_data, map_type="R", rule=rule)

        players = [player for file_name in file_names for player in get_players_of_a_game(path_data, file_name)]
        for player in players:
            player.classify_player(u1_def_neu, u1_neu_col)

        values = np.unique(list(BINNING_DICT.values()))
        stars = np.arange(6)

        probas_col, _ = bootstrap(values, stars, players, "collaborator", bootstrap_reps)
        probas_neu, _ = bootstrap(values, stars, players, "neutral", bootstrap_reps)
        probas_def, _ = bootstrap(values, stars, players, "defector", bootstrap_reps)

        for i, P in ((0, "P0"), (1, "P1234"), (5, "P5")):
            np.savetxt(
                path_data_figures + f"exp/classification/rule_{rule}/P{P}_col.txt",
                np.column_stack(
                    (
                        values,
                        probas_col["mean"][:, i],
                        probas_col["err"][:, :, i].T,
                    )
                ),
                fmt=("%d", "%f", "%f", "%f"),
            )
            np.savetxt(
                path_data_figures + f"exp/classification/rule_{rule}/P{P}_neu.txt",
                np.column_stack(
                    (
                        values,
                        probas_neu["mean"][:, i],
                        probas_neu["err"][:, :, i].T,
                    )
                ),
                fmt=("%d", "%f", "%f", "%f"),
            )
            np.savetxt(
                path_data_figures + f"exp/classification/rule_{rule}/P{P}_def.txt",
                np.column_stack(
                    (
                        values,
                        probas_def["mean"][:, i],
                        probas_def["err"][:, :, i].T,
                    )
                ),
                fmt=("%d", "%f", "%f", "%f"),
            )


def bootstrap(values, stars, players, player_type, bootstrap_reps):
    bs_proba = []
    bs_popt = []
    for _ in range(bootstrap_reps):
        bs_players = np.random.choice(players, replace=True, size=len(players))
        numbers_of_stars_played = binning(
            get_numbers_of_stars_played((player for player in bs_players if player.type == player_type)),
            BINNING_DICT,
        )
        probas = get_p0_p5_fit(values, stars, numbers_of_stars_played)
        bs_proba.append(probas)
        if player_type == "neutral":
            bs_popt.append(get_popt_model(values, probas, "linear"))
        else:
            bs_popt.append(get_popt_model(values, probas, "tanh"))
    bs_popt = np.array(bs_popt, dtype=float)
    bs_proba = np.array(bs_proba, dtype=float)
    return get_mean_err(bs_proba), get_mean_err(bs_popt)


def get_mean_err(bs_list):
    dict_ = {}
    dict_["mean"] = np.nanmean(bs_list, axis=0)
    dict_["err"] = np.abs(np.nanpercentile(bs_list, [50 - 34.13, 50 + 34.13], axis=0) - dict_["mean"])
    return dict_


def get_numbers_of_stars_played(players):
    numbers_of_stars_played = defaultdict(list)
    for player in players:
        for value, stars in player.numbers_of_stars_played.items():
            numbers_of_stars_played[value] += stars
    return numbers_of_stars_played


def binning(dict_, binning_dict):
    new_dict = defaultdict(list)
    for value, new_value in binning_dict.items():
        new_dict[new_value] += dict_[value]
    return new_dict


def p0_p5_distribution(stars, p0, p5):
    p1 = p2 = p3 = p4 = (1 - p0 - p5) / 4
    return np.array([p0, p1, p2, p3, p4, p5])


def p0_p5_distribution_same_mean(mean):
    def f(stars, p0):
        distribution = p0_p5_distribution(stars, p0, 2 / 5 * mean + p0 - 1)
        return distribution

    return f


def get_p0_p5_fit(values, stars, numbers_of_stars_played):
    probas = []
    for value in values:
        if nspv := numbers_of_stars_played[value]:
            mean = np.mean(nspv)
            f = p0_p5_distribution_same_mean(mean)
            y_data = [nspv.count(star) / len(nspv) for star in stars]
            bounds = (max(0, 1 - 2 / 5 * mean - 1e-10), min(1, 2 * (1 - mean / 5) + 1e-10))  # p0, p5 in [0, 1]
            p0 = min(1, curve_fit(f, stars, y_data, bounds=bounds)[0][0])
            p5 = max(0, 2 / 5 * mean + p0 - 1)  # constant mean
            p1 = p2 = p3 = p4 = max(0, (1 - p0 - p5) / 4)  # sum equal to 1
            probas.append([p0, p1, p2, p3, p4, p5])
        else:
            probas.append([None] * 6)
    return np.array(probas)


def linear_function(x, u0):
    return np.full(x.shape, u0)


def tanh_function(x, u0, u1):
    return (1 + np.tanh((x - u0) * u1 / 99)) / 2


def get_probas_fit_tanh(values_without_None, probas_without_None):
    try:
        return curve_fit(tanh_function, values_without_None, probas_without_None, p0=(10, 0.1))
    except RuntimeError:
        return curve_fit(tanh_function, values_without_None, probas_without_None, p0=(10, -11))


def get_popt_model(values, probas, function):

    values_without_None = []
    probas_without_None = []
    for value, param in zip(values, probas):
        if not (param[0] is None or np.isnan(param[0])):
            values_without_None.append(value)
            probas_without_None.append(param)
    values_without_None = np.array(values_without_None)
    probas_without_None = np.array(probas_without_None)

    if function == "tanh":
        popt_0, _ = get_probas_fit_tanh(values_without_None, probas_without_None[:, 0])
        popt_5, _ = get_probas_fit_tanh(values_without_None, probas_without_None[:, 5])
    else:
        popt_0, _ = curve_fit(linear_function, values_without_None, probas_without_None[:, 0])
        popt_5, _ = curve_fit(linear_function, values_without_None, probas_without_None[:, 5])

    return popt_0.tolist() + popt_5.tolist()


def get_probas_model(values_model, popt, function):
    N = int(len(popt) / 2)
    probas_model = [0] * 6
    if function == "tanh":
        probas_model[0] = tanh_function(values_model, *popt[:N])
        probas_model[5] = tanh_function(values_model, *popt[N:])
    if function == "linear":
        probas_model[0] = linear_function(values_model, *popt[:N])
        probas_model[5] = linear_function(values_model, *popt[N:])
    probas_model[1] = probas_model[2] = probas_model[3] = probas_model[4] = (1 - probas_model[0] - probas_model[5]) / 4
    return np.array(probas_model).T


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 1000

    main(path_data, path_data_figures, bootstrap_reps)
