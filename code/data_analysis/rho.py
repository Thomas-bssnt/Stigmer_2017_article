import numpy as np
from scipy.optimize import curve_fit

from modules.game import Game
from modules.gameset import GameSet


def main(path_data, path_data_figures, rule, bootstrap_reps):

    game = Game(path_data, "S01-A1-R1-MR-01")
    N_v = [0] * (max(game.V) + 1)
    for value, count in zip(*np.unique(game.V, return_counts=True)):
        N_v[value] = count

    games = GameSet(path_data, map_type="R", rule=rule)

    rho_mean, rho_err, eps_mean, eps_err, alpha_mean, alpha_err = bootstrap(games, bootstrap_reps, N_v)
    print(f"Rule = {rule}, esp = {eps_mean} ± {np.mean(eps_err)}, alpha = {alpha_mean} ± {np.mean(alpha_err)}")

    np.savetxt(
        path_data_figures + f"exp/observables/rule_{rule}/rho.txt",
        np.column_stack(
            (
                np.arange(len(rho_mean)),
                rho_mean,
                rho_err.T,
            )
        ),
        fmt=("%d", "%f", "%f", "%f"),
    )

    np.savetxt(
        path_data_figures + f"exp/observables/rule_{rule}/rho_fit.txt",
        np.column_stack(
            (
                np.arange(100),
                rho_fit(eps_mean, alpha_mean, N_v),
            )
        ),
        fmt=("%d", "%f"),
    )


def rho_fit(epsilon, alpha, N_v):
    return epsilon / np.sum(N_v) + (1 - epsilon) * np.arange(len(N_v)) ** alpha / np.nansum(
        np.arange(len(N_v)) ** alpha * N_v
    )


def fit_function(N_v):
    def rho_fit(V, epsilon, alpha):
        return np.log10(epsilon / np.sum(N_v) + (1 - epsilon) * V**alpha / np.nansum(V**alpha * N_v))

    return rho_fit


def bootstrap(games, bootstrap_reps, N_v):

    bs_rho = []
    bs_eps = []
    bs_alpha = []

    for _ in range(bootstrap_reps):

        occurrences_stars = [0] * len(N_v)
        numberStars = 0
        for game in np.random.choice(list(games), replace=True, size=len(games)):
            for value, stars in zip(game.V, np.sum(game.stars_played, axis=0)):
                occurrences_stars[value] += stars
                numberStars += stars

        rho = np.array(occurrences_stars) / N_v / numberStars
        V_v = [value for value, p in enumerate(rho) if not np.isnan(p)]
        N_v_ = [n for n in N_v if n != 0]
        rho_log = [np.log10(p) for p in rho if not np.isnan(p)]
        popts, _ = curve_fit(fit_function(N_v_), V_v, rho_log, p0=[0.5, 2], bounds=([0, 0], [1, np.inf]))

        bs_rho.append(rho)
        bs_eps.append(popts[0])
        bs_alpha.append(popts[1])

    rho_mean = list(np.nanmean(bs_rho, axis=0))
    rho_err = np.abs(np.percentile(bs_rho, [50 - 34.13, 50 + 34.13], axis=0) - rho_mean)
    eps_mean = np.mean(bs_eps)
    eps_err = np.abs(np.percentile(bs_eps, [50 - 34.13, 50 + 34.13], axis=0) - eps_mean)
    alpha_mean = np.mean(bs_alpha)
    alpha_err = np.abs(np.percentile(bs_alpha, [50 - 34.13, 50 + 34.13], axis=0) - alpha_mean)

    return rho_mean, rho_err, eps_mean, eps_err, alpha_mean, alpha_err


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 1000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
