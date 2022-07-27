from collections import defaultdict
from json import load

import numpy as np

from modules.files import condition_true, get_filenames


class Game:
    def __init__(self, path_data, filename):
        # In data
        path_in_file = path_data + f"/session_{filename[1:3]}/in/{filename}.json"
        with open(path_in_file) as in_file:
            self.in_data = load(in_file)
        self.map = np.array(self.in_data["map"]).flatten()

        # Out data
        path_out_file = path_data + f"/session_{filename[1:3]}/out/{filename}.csv"
        out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)
        self.cells_played = np.zeros((self.in_data["numberRounds"], self.in_data["mapSize"] ** 2), dtype=int)
        self.stars_played = np.zeros((self.in_data["numberRounds"], self.in_data["mapSize"] ** 2), dtype=int)
        self.scores = defaultdict(int)
        self.pseudo_scores = defaultdict(int)
        for round_, playerId, mapX, mapY, value, numberStars, score in out_data:
            self.cells_played[round_ - 1, mapY * self.in_data["mapSize"] + mapX] += 1
            self.stars_played[round_ - 1, mapY * self.in_data["mapSize"] + mapX] += numberStars
            self.scores[playerId] += score
            self.pseudo_scores[playerId] += value

        # Observables
        self.observables = None

    def __repr__(self):
        return (
            f"(S{self.in_data['sessionNumber']} "
            f"{self.in_data['groupId']}{self.in_data['gameNumber']} "
            f"R{self.in_data['ruleNumber']} "
            f"M{self.in_data['mapType']}-{self.in_data['mapNumber']})"
        )

    def compute_observables(self):
        p_c = _safe_normalization(self.stars_played)
        P_c = _safe_normalization(np.cumsum(self.stars_played, axis=0))
        q_c = _safe_normalization(self.cells_played)
        Q_c = _safe_normalization(np.cumsum(self.cells_played, axis=0))

        self.observables = {}

        V_max_3, V_max_2, V_max_1 = np.sort(self.map)[-3:]
        self.observables["p_"] = self._get_performance(p_c, V_max_1)
        self.observables["P"] = self._get_performance(P_c, V_max_1)
        self.observables["q_"] = self._get_performance(q_c, (V_max_1 + V_max_2 + V_max_3) / 3)
        self.observables["Q"] = self._get_performance(Q_c, (V_max_1 + V_max_2 + V_max_3) / 3)

        self.observables["IPR_p_"] = self._get_IPR(p_c)
        self.observables["IPR_P"] = self._get_IPR(P_c)
        self.observables["IPR_q_"] = self._get_IPR(q_c)
        self.observables["IPR_Q"] = self._get_IPR(Q_c)

        self.observables["F_P"] = self._get_fidelity(P_c)
        self.observables["F_Q"] = self._get_fidelity(Q_c)

    def _get_performance(self, array, denominator):
        return np.sum(array * self.map, axis=1) / denominator

    @staticmethod
    def _get_IPR(array):
        return 1 / np.sum(array**2, axis=1)

    def _get_fidelity(self, array):
        return np.sum(np.sqrt(array * self.map / np.sum(self.map)), axis=1)


class AggregatedGame:
    def __init__(self, games):
        games = list(games)

        # Check if all games are using random maps (i.e. the maps have the exact same values)
        if any(game.in_data["mapType"] != "R" for game in games):
            raise ValueError("Only work for random maps.")

        # Check if all games shared the same rule
        if len({game.in_data["ruleNumber"] for game in games}) != 1:
            raise ValueError("All games to not share the same rule.")

        # In_data
        self.in_data = {}
        self.in_data["ruleNumber"] = games[0].in_data["ruleNumber"]
        self.in_data["mapType"] = games[0].in_data["mapType"]
        self.V_v, self.N_v = np.unique(games[0].map, return_counts=True)

        # Out_data
        i_from_v = {v: list(self.V_v).index(v) for v in games[0].map}
        self.stars_played_values = np.zeros((len(games[0].stars_played), len(self.V_v)), dtype=int)
        self.cells_played_values = np.zeros((len(games[0].stars_played), len(self.V_v)), dtype=int)
        for game in games:
            for round_ in range(len(game.stars_played)):
                for value, star, cell in zip(game.map, game.stars_played[round_], game.cells_played[round_]):
                    self.stars_played_values[round_][i_from_v[value]] += star
                    self.cells_played_values[round_][i_from_v[value]] += cell

        # Observables
        self.observables = None

    def __repr__(self):
        return f"(R{self.in_data['ruleNumber']} M{self.in_data['mapType']})"

    def compute_observables(self):
        p_v = _safe_normalization(self.stars_played_values) / self.N_v
        P_v = _safe_normalization(np.cumsum(self.stars_played_values, axis=0)) / self.N_v
        q_v = _safe_normalization(self.cells_played_values) / self.N_v
        Q_v = _safe_normalization(np.cumsum(self.cells_played_values, axis=0)) / self.N_v

        self.observables = {}

        V_max_3, V_max_2, V_max_1 = np.sort(self.V_v)[-3:]
        self.observables["p_"] = self._get_performance(p_v, V_max_1)
        self.observables["P"] = self._get_performance(P_v, V_max_1)
        self.observables["q_"] = self._get_performance(q_v, (V_max_1 + V_max_2 + V_max_3) / 3)
        self.observables["Q"] = self._get_performance(Q_v, (V_max_1 + V_max_2 + V_max_3) / 3)

        self.observables["IPR_p_"] = self._get_IPR(p_v)
        self.observables["IPR_P"] = self._get_IPR(P_v)
        self.observables["IPR_q_"] = self._get_IPR(q_v)
        self.observables["IPR_Q"] = self._get_IPR(Q_v)

        self.observables["F_P"] = self._get_fidelity(P_v)
        self.observables["F_Q"] = self._get_fidelity(Q_v)

    def _get_performance(self, array, denominator):
        return np.sum(self.N_v * array * self.V_v, axis=1) / denominator

    def _get_IPR(self, array):
        return 1 / np.sum(self.N_v * array**2, axis=1)

    def _get_fidelity(self, array):
        return 1 - np.sum(self.N_v * np.sqrt(array * self.V_v / np.sum(self.N_v * self.V_v)), axis=1)


class GameSet(set):
    def __init__(
        self,
        path_data,
        session=None,
        group=None,
        game_number=None,
        rule=None,
        map_type=None,
        map_number=None,
    ):
        super().__init__()
        for filename in get_filenames(
            path_data,
            session=session,
            group=group,
            game_number=game_number,
            rule=rule,
            map_type=map_type,
            map_number=map_number,
        ):
            self.add(Game(path_data=path_data, filename=filename))

    def subset(self, session=None, group=None, gameNumber=None, rule=None, map_type=None, map_number=None):
        for game in self:
            if (
                condition_true(game.in_data["sessionNumber"], session)
                and condition_true(game.in_data["groupId"], group)
                and condition_true(game.in_data["gameNumber"], gameNumber)
                and condition_true(game.in_data["ruleNumber"], rule)
                and condition_true(game.in_data["mapType"], map_type)
                and condition_true(game.in_data["mapNumber"], map_number)
            ):
                yield game

    def compute_observables(self):
        for game in self:
            game.compute_observables()


def _safe_normalization(array):
    normalized_array = np.zeros(array.shape)
    for round_ in range(len(array)):
        if (sum_ := np.sum(array[round_])) != 0:
            normalized_array[round_] = array[round_] / sum_
        else:
            normalized_array[round_] = 1 / len(array[0])
    return normalized_array


if __name__ == "__main__":

    path_data = "./data/"

    print(GameSet(path_data))
    print(GameSet(path_data, session=1))
    print(GameSet(path_data, session=[1, 5]))
    print(GameSet(path_data, session={1, 5}))
    print(GameSet(path_data, session=range(11)))

    A = GameSet(path_data)
    print(A)
    print(list(A.subset(rule=1, map_type="R", map_number=4)))

    B = GameSet(path_data, map_type="R", rule=2)
    print(B)

    C = AggregatedGame(B)
    print(C)
