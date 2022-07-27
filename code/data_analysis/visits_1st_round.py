import numpy as np

from modules.files import get_filenames


def main(path_data, path_data_figures):

    file_names = get_filenames(path_data)

    map_visits = np.zeros((3, 15, 15))

    for file_name in file_names:

        path_out_file = path_data + f"/session_{file_name[1:3]}/out/{file_name}.csv"
        out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)

        visits_round = [[] for _ in range(5)]
        for round_, playerId, mapX, mapY, _, _, _ in out_data:
            if round_ == 1:
                visits_round[int(playerId[1]) - 1].append([mapX, mapY])

        for visits_round_player in visits_round:
            for i, (mapX, mapY) in enumerate(visits_round_player):
                map_visits[i, mapY, mapX] += 1

    np.savetxt(path_data_figures + "exp/other/cell_1.txt", map_visits[0])
    np.savetxt(path_data_figures + "exp/other/cell_2.txt", map_visits[1])
    np.savetxt(path_data_figures + "exp/other/cell_3.txt", map_visits[2])


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    main(path_data, path_data_figures)
    main(path_data, path_data_figures)
