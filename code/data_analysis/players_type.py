import numpy as np

from modules.files import get_filenames
from modules.player import get_players_of_a_game


def main(path_data, path_data_figures, rule):

    u1_def_neu, u1_neu_col = np.genfromtxt(path_data_figures + "exp/classification/thresholds.txt")

    file_names = get_filenames(path_data, map_type="R", rule=rule)

    players = [player for file_name in file_names for player in get_players_of_a_game(path_data, file_name)]
    for player in players:
        player.classify_player(u1_def_neu, u1_neu_col)

    players.sort(key=lambda player: -player.u1)

    with open(path_data_figures + f"exp/classification/rule_{rule}/players_u0_u1.txt", "w") as file:
        for player in players:
            file.write(f"{player.u0} {player.u1}\n")


if __name__ == "__main__":

    path_data = "./data/"
    path_data_figures = "./data_figures/"

    main(path_data, path_data_figures, 1)
    main(path_data, path_data_figures, 2)
