from json import load

import numpy as np

from modules.files import get_filenames


def main(path_data, path_data_figures, rule, values, bootstrap_reps):

    if isinstance(values, int):
        values = [values]

    file_names = get_filenames(path_data, map_type="R", rule=rule)

    probability_finding_value = []
    for file_name in file_names:
        for value in values:
            probability_finding_value += get_probability_finding_value(path_data, file_name, value)

    mean, err = bootstrap(probability_finding_value, bootstrap_reps)

    np.savetxt(
        path_data_figures + f"exp/observables/rule_{rule}/proba_find_{'_'.join([str(value) for value in values])}.txt",
        np.column_stack(
            (
                np.arange(1, len(mean) + 1),
                mean,
                err.T,
            )
        ),
        fmt=("%d", "%f", "%f", "%f"),
    )


def get_probability_finding_value(path_data, file_name, value):

    path_in_file = path_data + f"/session_{file_name[1:3]}/in/{file_name}.json"
    with open(path_in_file) as in_file:
        in_data = load(in_file)

    path_out_file = path_data + f"/session_{file_name[1:3]}/out/{file_name}.csv"
    out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)

    findings = {}
    for mapY in range(in_data["mapSize"]):
        for mapX in range(in_data["mapSize"]):
            if in_data["map"][mapY][mapX] == value:
                findings[(mapX, mapY)] = {}

    for round_, playerId, mapX, mapY, v, _, _ in out_data:
        if v == value:
            if playerId not in findings[(mapX, mapY)]:
                findings[(mapX, mapY)][playerId] = round_

    probas = []
    for findings_cell in findings.values():
        probability_finding_value = np.zeros(in_data["numberRounds"])
        for round_ in findings_cell.values():
            probability_finding_value[round_ - 1 :] += 1
        probability_finding_value /= in_data["numberPlayers"]
        probas.append(probability_finding_value)
    return probas


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

    main(path_data, path_data_figures, 1, 99, bootstrap_reps)
    main(path_data, path_data_figures, 1, [86, 85, 84], bootstrap_reps)
    main(path_data, path_data_figures, 1, [72, 71], bootstrap_reps)
    main(path_data, path_data_figures, 2, 99, bootstrap_reps)
    main(path_data, path_data_figures, 2, [86, 85, 84], bootstrap_reps)
    main(path_data, path_data_figures, 2, [72, 71], bootstrap_reps)
