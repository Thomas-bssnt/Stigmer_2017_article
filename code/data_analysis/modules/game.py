from collections import defaultdict
from json import load

import numpy as np


class Game:
    def __init__(self, path_data, filename):
        # In data
        path_in_file = path_data + f"/session_{filename[1:3]}/in/{filename}.json"
        with open(path_in_file) as in_file:
            self.in_data = load(in_file)

        # Out data
        path_out_file = path_data + f"/session_{filename[1:3]}/out/{filename}.csv"
        out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)
        self.cells_played = np.zeros((self.in_data["numberRounds"], self.in_data["mapSize"] ** 2), dtype=int)
        self.stars_played = np.zeros((self.in_data["numberRounds"], self.in_data["mapSize"] ** 2), dtype=int)
        self.scores = defaultdict(int)
        self.scores_R2 = defaultdict(int)
        self.cells_played_players = defaultdict(lambda: [[] for _ in range(self.in_data["numberRounds"])])
        for round_, playerId, mapX, mapY, value, numberStars, score in out_data:
            self.cells_played[round_ - 1, mapY * self.in_data["mapSize"] + mapX] += 1
            self.stars_played[round_ - 1, mapY * self.in_data["mapSize"] + mapX] += numberStars
            self.scores[playerId] += score
            self.scores_R2[playerId] += value
            self.cells_played_players[playerId][round_ - 1].append(value)

        # Observables
        self.observables = {}

        # Map values
        self.V = np.array(self.in_data["map"]).flatten()

        # Fractions
        q_c = self._get_fractions(self.cells_played)
        Q_c = self._get_fractions(np.cumsum(self.cells_played, axis=0))
        p_c = self._get_fractions(self.stars_played)
        P_c = self._get_fractions(np.cumsum(self.stars_played, axis=0))

        # Performance indicators
        V_max_3, V_max_2, V_max_1 = np.sort(self.V)[-3:]
        self.observables["q_"] = self._get_performance(q_c, self.V, (V_max_1 + V_max_2 + V_max_3) / 3)
        self.observables["Q"] = self._get_performance(Q_c, self.V, (V_max_1 + V_max_2 + V_max_3) / 3)
        self.observables["p_"] = self._get_performance(p_c, self.V, V_max_1)
        self.observables["P"] = self._get_performance(P_c, self.V, V_max_1)

        # Inverse Participation ratio
        self.observables["IPR_q_"] = self._get_IPR(q_c)
        self.observables["IPR_Q"] = self._get_IPR(Q_c)
        self.observables["IPR_p_"] = self._get_IPR(p_c)
        self.observables["IPR_P"] = self._get_IPR(P_c)

        # Fidelity
        self.observables["F_Q"] = self._get_fidelity(Q_c, self.V)
        self.observables["F_P"] = self._get_fidelity(P_c, self.V)

        # Value best cells
        (
            self.observables["V3"],
            self.observables["V2"],
            self.observables["V1"],
        ) = self._get_value_best_cells(self.cells_played_players)

    def __repr__(self):
        return (
            f"(S{self.in_data['sessionNumber']} "
            f"{self.in_data['groupId']}{self.in_data['gameNumber']} "
            f"R{self.in_data['ruleNumber']} "
            f"M{self.in_data['mapType']}-{self.in_data['mapNumber']})"
        )

    @staticmethod
    def _get_fractions(array):
        sum_ = np.sum(array, axis=1)
        return np.divide(array.T, sum_, where=sum_ != 0).T

    @staticmethod
    def _get_performance(array, V, denominator):
        return np.sum(array * V, axis=1) / denominator

    @staticmethod
    def _get_IPR(array):
        sum_ = np.sum(array**2, axis=1)
        return np.divide(1, sum_, out=np.zeros(sum_.shape), where=sum_ != 0)

    @staticmethod
    def _get_fidelity(array, V):
        return np.sum(np.sqrt(array * V / np.sum(V)), axis=1)

    @staticmethod
    def _get_value_best_cells(cells_played_players):
        return np.mean(
            [np.sort(cells_played_player) for cells_played_player in cells_played_players.values()], axis=0
        ).T


if __name__ == "__main__":

    path_data = "./data/"

    game = Game(path_data, "S01-A1-R1-MR-01")

    print(game)
