import numpy as np

from modules.game import GameSet


def main(path_data, path_data_figures, rule, bootstrap_reps):

    map_type = "R"

    games = GameSet(path_data, map_type=map_type, rule=rule)
    games.compute_observables()

    for observable in list(games)[0].observables.keys():

        observable_list = [game.observables[observable] for game in games]
        mean, err = bootstrap(observable_list, bootstrap_reps)

        np.savetxt(
            path_data_figures + f"exp/cell/rule_{rule}/{observable}.txt",
            np.column_stack(
                (
                    np.arange(1, len(mean) + 1),
                    mean,
                    err.T,
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

    bootstrap_reps = 1000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
