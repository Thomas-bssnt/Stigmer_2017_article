from collections import defaultdict

import numpy as np

from modules.gameset import GameSet


def main(path_data, path_data_figures, rule, bootstrap_reps):
    games = GameSet(path_data, map_type="R", rule=rule)
    observables = bootstrap(list(games), bootstrap_reps)
    for name, (mean, err) in observables.items():
        np.savetxt(
            path_data_figures + f"exp/observables/rule_{rule}/{name}.txt",
            np.column_stack(
                (
                    np.arange(1, len(mean) + 1),
                    mean,
                    err.T,
                )
            ),
            fmt=("%d", "%f", "%f", "%f"),
        )


def bootstrap(games, bootstrap_reps):
    bs_observables = defaultdict(list)
    for _ in range(bootstrap_reps):
        bs_games = np.random.choice(games, replace=True, size=len(games))
        for observable in bs_games[0].observables.keys():
            bs_observables[observable].append(np.mean([game.observables[observable] for game in bs_games], axis=0))
    return {
        observable: (
            np.mean(bs_observables[observable], axis=0),
            np.abs(
                np.percentile(bs_observables[observable], [50 - 34.13, 50 + 34.13], axis=0)
                - np.mean(bs_observables[observable], axis=0)
            ),
        )
        for observable in bs_observables
    }


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    bootstrap_reps = 10000

    main(path_data, path_data_figures, 1, bootstrap_reps)
    main(path_data, path_data_figures, 2, bootstrap_reps)
