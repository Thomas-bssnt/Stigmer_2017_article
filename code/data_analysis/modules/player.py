from json import load

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import rankdata


class Player:
    def __init__(self, session, group, game, rule, map_type, numberRounds, playerId):

        # Game information
        self.session = session
        self.group = group
        self.game = game
        self.rule = rule
        self.map_type = map_type

        # self information
        self.playerId = playerId

        # What he played (initialization)
        self.indices = [[] for _ in range(numberRounds)]
        self.visits = [[] for _ in range(numberRounds)]
        self.stars = [[] for _ in range(numberRounds)]

        # Global performance (initialization)
        self.score = 0
        self.rank = None

    def __repr__(self):
        return f"S{self.session}-{self.group}{self.game}-R{self.rule}-M{self.map_type}-P{self.playerId}"

    def classify_player(self, u1_def_neu, u1_neu_col):

        # Fit
        (self.u0, self.u1), _ = curve_fit(linear_function, self.values, self.mean_number_of_stars_played, p0=[2.5, 0])

        # Error to the fit
        self.sigma = 0
        for value, mean_number_of_stars in zip(self.values, self.mean_number_of_stars_played):
            self.sigma += (mean_number_of_stars - linear_function(value, self.u0, self.u1)) ** 2

        # Characterization of the type
        if self.u1 > u1_neu_col:
            self.type = "collaborator"
        elif self.u1 < u1_def_neu:
            self.type = "defector"
        else:
            self.type = "neutral"


def get_players_of_a_game(path_data, file_name):

    # Import the data of the game
    path_in_file = path_data + f"/session_{file_name[1:3]}/in/{file_name}.json"
    with open(path_in_file) as in_file:
        in_data = load(in_file)

    path_out_file = path_data + f"/session_{file_name[1:3]}/out/{file_name}.csv"
    out_data = np.genfromtxt(path_out_file, dtype=None, delimiter=",", names=True, encoding=None)

    # Create the players
    players = [
        Player(
            in_data["sessionNumber"],
            in_data["groupId"],
            in_data["gameNumber"],
            in_data["ruleNumber"],
            in_data["mapType"],
            in_data["numberRounds"],
            playerId,
        )
        for playerId in range(1, in_data["numberPlayers"] + 1)
    ]

    # Add the data of the game to the players
    for round_, playerId, mapX, mapY, value, numberStars, score in out_data:
        players[int(playerId[1]) - 1].indices[round_ - 1].append(mapY * in_data["mapSize"] + mapX)
        players[int(playerId[1]) - 1].visits[round_ - 1].append(value)
        players[int(playerId[1]) - 1].stars[round_ - 1].append(numberStars)
        if in_data["ruleNumber"] == 1:
            players[int(playerId[1]) - 1].score += value
        else:
            players[int(playerId[1]) - 1].score += score

    # Get the rank of the players
    scores = np.array([player.score for player in players])
    for player, rank in zip(players, rankdata(-scores, method="min")):
        player.rank = rank

    # Get observables about the players visits and stars
    for player in players:
        player.values = np.unique(player.visits)
        player.numbers_of_stars_played = {value: [] for value in player.values}
        for iRound in range(len(player.visits)):
            for iTurn in range(len(player.visits[0])):
                player.numbers_of_stars_played[player.visits[iRound][iTurn]].append(player.stars[iRound][iTurn])
        player.mean_number_of_stars_played = np.array(
            [np.mean(player.numbers_of_stars_played[value]) for value in player.values]
        )
        player.number_of_visits = np.array([len(player.numbers_of_stars_played[value]) for value in player.values])

    return players


def linear_function(x, u0, u1):
    return u0 + 5 * u1 * x / 99
