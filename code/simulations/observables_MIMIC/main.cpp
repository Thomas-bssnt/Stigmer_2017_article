#include <algorithm>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "Game.h"
#include "Player.h"
#include "useful_functions.h"

int main()
{
    // Parameters of the program
    const int numberOfGames{10000};

    // Parameters of the game
    const int numberOfRounds{20};
    const int numberOfPlayers{5};
    const int numberOfValues{100};
    std::vector<Rule> rules{Rule::rule_1, Rule::rule_2};

    // Path of the in and out files
    const std::string pathDataFigures{"../../../data_figures/"};
    const std::string pathParameters{pathDataFigures + "model/parameters/"};
    const std::string pathObservables{pathDataFigures + "model/"};

    for (auto &rule : rules)
    {
        // Import the optimized parameters
        const std::string ruleStr{stringFromEnum(rule)};
        const std::vector<double> parametersVisits{getParameters(pathParameters + ruleStr + "cells.txt")};
        const std::vector<double> parametersPlayersType{getParameters(pathParameters + ruleStr + "players_type.txt")};
        const std::vector<double> parametersStarsCol{getParameters(pathParameters + "stars_col.txt")};
        const std::vector<double> parametersStarsNeu{getParameters(pathParameters + "stars_neu.txt")};
        const std::vector<double> parametersStarsDef{getParameters(pathParameters + "stars_def.txt")};

        // Initialization of the observables
        std::vector<std::vector<double>> Q(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> q(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> P(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> p(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> IPR_Q(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> IPR_q(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> IPR_P(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> IPR_p(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> F_Q(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> F_P(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> B1(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> B2(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> B3(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> V1(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> V2(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> V3(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> VB1(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> VB2(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<std::vector<double>> VB3(numberOfRounds, std::vector<double>(numberOfGames, 0));
        std::vector<double> S(numberOfGames * numberOfPlayers, 0);
        std::vector<double> S_team(numberOfGames, 0);
        std::vector<int> N_def(numberOfGames, 0);
        std::vector<int> rankCollaborator(numberOfPlayers, 0);
        std::vector<int> rankNeutral(numberOfPlayers, 0);
        std::vector<int> rankDefector(numberOfPlayers, 0);
        std::vector<int> find99(numberOfRounds, 0);
        std::vector<int> find80(numberOfRounds, 0);
        std::vector<int> find70(numberOfRounds, 0);

        // Loop on the games
#pragma omp parallel for
        for (int iGame = 0; iGame < numberOfGames; ++iGame)
        {
            // Creation of the game
            Game game(numberOfRounds, numberOfPlayers, rule);

            // Creation of the players
            std::vector<Player> players;
            for (int iPlayer{0}; iPlayer < numberOfPlayers; ++iPlayer)
            {
                switch (getPlayerType(parametersPlayersType))
                {
                case PlayerType::collaborator:
                    players.emplace_back(iPlayer, game.getAddress(), parametersVisits, parametersStarsCol, PlayerType::collaborator);
                    break;
                case PlayerType::neutral:
                    players.emplace_back(iPlayer, game.getAddress(), parametersVisits, parametersStarsNeu, PlayerType::neutral);
                    break;
                case PlayerType::defector:
                    players.emplace_back(iPlayer, game.getAddress(), parametersVisits, parametersStarsDef, PlayerType::defector);
                    ++N_def[iGame];
                    break;
                default:
                    break;
                }
            }

            // Compute some properties of the game
            std::vector<int> sortedValues{game.getValues()};
            std::sort(sortedValues.begin(), sortedValues.end(), std::greater<>());
            const int Vmax1{sortedValues[0]};
            const int Vmax2{sortedValues[1]};
            const int Vmax3{sortedValues[2]};
            const int Smax{getScoreMax(rule, numberOfRounds, Vmax1, Vmax2, Vmax3)};

            // Play the game
            for (int iRound{0}; iRound < numberOfRounds; ++iRound)
            {
                // Do one Round
                for (auto &player : players)
                {
                    player.playARound();

                    const std::vector<bool> foundBestCells{player.getFoundBestCells()};
                    // Found best cell
                    find99[iRound] += foundBestCells[0];
                    find80[iRound] += foundBestCells[1];
                    find80[iRound] += foundBestCells[2];
                    find80[iRound] += foundBestCells[3];
                    find80[iRound] += foundBestCells[4];
                    find70[iRound] += foundBestCells[5];
                    find70[iRound] += foundBestCells[6];
                    find70[iRound] += foundBestCells[7];
                    find70[iRound] += foundBestCells[8];
                    // Update the values of VB1, VB2, VB3
                    std::vector<int> mapValues{99, 86, 86, 85, 84, 72, 72, 71, 71, 53, 53, 53, 51, 46, 45,
                                               45, 44, 44, 44, 43, 43, 28, 27, 27, 27, 24, 24, 24, 22, 21,
                                               20, 20, 20, 20, 20, 19, 19, 14, 14, 13, 13, 13, 12, 12, 11,
                                               9, 9, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6,
                                               6, 6, 6, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
                                               4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                               3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                               2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                               1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    int i{0};
                    int iCell{0};
                    while (i < 3)
                    {
                        if (foundBestCells[iCell])
                        {
                            if (i == 0)
                            {
                                VB1[iRound][iGame] += mapValues[iCell];
                            }
                            else if (i == 1)
                            {
                                VB2[iRound][iGame] += mapValues[iCell];
                            }
                            else if (i == 2)
                            {
                                VB3[iRound][iGame] += mapValues[iCell];
                            }
                            ++i;
                        }
                        ++iCell;
                    }
                }

                // Update some of the observables
                Q[iRound][iGame] = computeQ(game.getVisitsDistribution(), game.getValues(), Vmax1, Vmax2, Vmax3);
                q[iRound][iGame] = computeQ(game.getInstantaneousVisitsDistribution(), game.getValues(), Vmax1, Vmax2, Vmax3);
                P[iRound][iGame] = computeP(game.getStarsDistribution(), game.getValues(), Vmax1);
                p[iRound][iGame] = computeP(game.getInstantaneousStarsDistribution(), game.getValues(), Vmax1);
                IPR_Q[iRound][iGame] = computeIPR(game.getVisitsDistribution());
                IPR_q[iRound][iGame] = computeIPR(game.getInstantaneousVisitsDistribution());
                IPR_P[iRound][iGame] = computeIPR(game.getStarsDistribution());
                IPR_p[iRound][iGame] = computeIPR(game.getInstantaneousStarsDistribution());
                F_Q[iRound][iGame] = computeF(game.getVisitsDistribution(), game.getValues());
                F_P[iRound][iGame] = computeF(game.getStarsDistribution(), game.getValues());
            }

            for (int iPlayer{0}; iPlayer < numberOfPlayers; ++iPlayer)
            {
                // Update the values of B1, B2, B3
                const std::vector<std::vector<int>> playBestCellsRound{players[iPlayer].getPlayBestCellsRound()};
                for (int iRound{0}; iRound < numberOfRounds; ++iRound)
                {
                    B1[iRound][iGame] += playBestCellsRound[0][iRound];
                    B2[iRound][iGame] += playBestCellsRound[1][iRound];
                    B3[iRound][iGame] += playBestCellsRound[2][iRound];
                }
                // Update the values of V1, V2, V3
                const std::vector<std::vector<int>> valueBestCells{players[iPlayer].getValueBestCells()};
                for (int iRound{0}; iRound < numberOfRounds; ++iRound)
                {
                    V1[iRound][iGame] += valueBestCells[0][iRound];
                    V2[iRound][iGame] += valueBestCells[1][iRound];
                    V3[iRound][iGame] += valueBestCells[2][iRound];
                }
                // Update the scores
                const double normalizedScore{game.getScoreOfPlayer(iPlayer) / static_cast<double>(Smax)};
                S[iPlayer + numberOfPlayers * iGame] = normalizedScore;
                S_team[iGame] += normalizedScore;
            }
            for (int iRound{0}; iRound < numberOfRounds; ++iRound)
            {
                B1[iRound][iGame] /= numberOfPlayers;
                B2[iRound][iGame] /= numberOfPlayers;
                B3[iRound][iGame] /= numberOfPlayers;
                V1[iRound][iGame] /= numberOfPlayers;
                V2[iRound][iGame] /= numberOfPlayers;
                V3[iRound][iGame] /= numberOfPlayers;
                VB1[iRound][iGame] /= numberOfPlayers;
                VB2[iRound][iGame] /= numberOfPlayers;
                VB3[iRound][iGame] /= numberOfPlayers;
            }
            S_team[iGame] /= numberOfPlayers;

            // Ranking
            std::vector<std::size_t> idx{sortIndexes(std::vector<double>(S.begin() + numberOfPlayers * iGame,
                                                                         S.begin() + numberOfPlayers * iGame + numberOfPlayers))};
            for (int iRank{0}; iRank < numberOfPlayers; ++iRank)
            {
                switch (players[idx[iRank]].getPlayerType())
                {
                case PlayerType::collaborator:
                    ++rankCollaborator[iRank];
                    break;
                case PlayerType::neutral:
                    ++rankNeutral[iRank];
                    break;
                case PlayerType::defector:
                    ++rankDefector[iRank];
                    break;
                default:
                    break;
                }
            }
        }

        std::vector<double> propCollaborator(numberOfPlayers, 0);
        std::vector<double> propNeutral(numberOfPlayers, 0);
        std::vector<double> propDefector(numberOfPlayers, 0);
        for (int iRank{0}; iRank < numberOfPlayers; ++iRank)
        {
            propCollaborator[iRank] = static_cast<double>(rankCollaborator[iRank]) / numberOfGames;
            propNeutral[iRank] = static_cast<double>(rankNeutral[iRank]) / numberOfGames;
            propDefector[iRank] = static_cast<double>(rankDefector[iRank]) / numberOfGames;
        }

        std::vector<double> probaFind99(numberOfRounds, 0.);
        std::vector<double> probaFind80(numberOfRounds, 0.);
        std::vector<double> probaFind70(numberOfRounds, 0.);
        for (int iRound{0}; iRound < numberOfRounds; ++iRound)
        {
            probaFind99[iRound] = static_cast<double>(find99[iRound]) / (numberOfPlayers * numberOfGames);
            probaFind80[iRound] = static_cast<double>(find80[iRound]) / (numberOfPlayers * numberOfGames * 4);
            probaFind70[iRound] = static_cast<double>(find70[iRound]) / (numberOfPlayers * numberOfGames * 4);
        }

        // Write the observables in different files
        saveObservable(pathObservables + "cell/" + ruleStr + "Q", getAverage2dVector(Q));
        saveObservable(pathObservables + "cell/" + ruleStr + "q_", getAverage2dVector(q));
        saveObservable(pathObservables + "cell/" + ruleStr + "P", getAverage2dVector(P));
        saveObservable(pathObservables + "cell/" + ruleStr + "p_", getAverage2dVector(p));
        saveObservable(pathObservables + "cell/" + ruleStr + "IPR_Q", getAverage2dVector(IPR_Q));
        saveObservable(pathObservables + "cell/" + ruleStr + "IPR_q_", getAverage2dVector(IPR_q));
        saveObservable(pathObservables + "cell/" + ruleStr + "IPR_P", getAverage2dVector(IPR_P));
        saveObservable(pathObservables + "cell/" + ruleStr + "IPR_p_", getAverage2dVector(IPR_p));
        saveObservable(pathObservables + "cell/" + ruleStr + "F_Q", getAverage2dVector(F_Q));
        saveObservable(pathObservables + "cell/" + ruleStr + "F_P", getAverage2dVector(F_P));
        saveObservable(pathObservables + "cell/" + ruleStr + "proba_find_99", probaFind99);
        saveObservable(pathObservables + "cell/" + ruleStr + "proba_find_86_85_84", probaFind80);
        saveObservable(pathObservables + "cell/" + ruleStr + "proba_find_72_71", probaFind70);
        saveObservable(pathObservables + "other/" + ruleStr + "B1", getAverage2dVector(B1));
        saveObservable(pathObservables + "other/" + ruleStr + "B2", getAverage2dVector(B2));
        saveObservable(pathObservables + "other/" + ruleStr + "B3", getAverage2dVector(B3));
        saveObservable(pathObservables + "other/" + ruleStr + "V1", getAverage2dVector(V1));
        saveObservable(pathObservables + "other/" + ruleStr + "V2", getAverage2dVector(V2));
        saveObservable(pathObservables + "other/" + ruleStr + "V3", getAverage2dVector(V3));
        saveObservable(pathObservables + "other/" + ruleStr + "VB1", getAverage2dVector(VB1));
        saveObservable(pathObservables + "other/" + ruleStr + "VB2", getAverage2dVector(VB2));
        saveObservable(pathObservables + "other/" + ruleStr + "VB3", getAverage2dVector(VB3));
        saveObservable(pathObservables + "other/" + ruleStr + "S", getHistogram(S, 50));
        saveObservable(pathObservables + "other/" + ruleStr + "S_team", getHistogram(S_team, 50));
        saveObservable(pathObservables + "other/" + ruleStr + "<S>", std::vector<double>{getAverage(S)});
        saveObservable(pathObservables + "other/" + ruleStr + "rk_col", propCollaborator);
        saveObservable(pathObservables + "other/" + ruleStr + "rk_neu", propNeutral);
        saveObservable(pathObservables + "other/" + ruleStr + "rk_def", propDefector);
        std::vector<double> mean(numberOfPlayers + 1, 0);
        std::vector<double> median(numberOfPlayers + 1, 0);
        getS_team_N_def(S_team, N_def, mean, median);
        saveObservable(pathObservables + "other/" + ruleStr + "S_team_N_def_mean", mean);
        saveObservable(pathObservables + "other/" + ruleStr + "S_team_N_def_median", median);
    }
    return 0;
}
