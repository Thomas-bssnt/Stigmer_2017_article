#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Game.h"
#include "MC_functions.h"
#include "Player.h"
#include "random.h"
#include "useful_functions.h"

double getAverageError(const int numberOfRounds, const int numberOfPlayers, const Rule &rule,
                       const std::vector<double> &parametersVisits, const std::vector<double> &parametersPlayersType,
                       const std::vector<double> &parametersStarsCol, const std::vector<double> &parametersStarsNeu,
                       const std::vector<double> &parametersStarsDef, const int numberOfGames,
                       const std::string &pathObservables)
{
    std::cerr << "ParametersVisits: ";
    for (const auto &parameter : parametersVisits)
    {
        std::cerr << parameter << " ";
    }
    std::cerr << "\n";

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
        }
        for (int iRound{0}; iRound < numberOfRounds; ++iRound)
        {
            B1[iRound][iGame] /= numberOfPlayers;
            B2[iRound][iGame] /= numberOfPlayers;
            B3[iRound][iGame] /= numberOfPlayers;
            V1[iRound][iGame] /= numberOfPlayers;
            V2[iRound][iGame] /= numberOfPlayers;
            V3[iRound][iGame] /= numberOfPlayers;
        }
    }

    // Compute the error
    const std::string ruleStr{stringFromEnum(rule)};
    double totalError{0.};
    totalError += computeError(pathObservables + "cell/" + ruleStr + "Q", getAverage2dVector(Q));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "q_", getAverage2dVector(q));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "P", getAverage2dVector(P));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "p_", getAverage2dVector(p));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "IPR_Q", getAverage2dVector(IPR_Q));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "IPR_q_", getAverage2dVector(IPR_q));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "IPR_P", getAverage2dVector(IPR_P));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "IPR_p_", getAverage2dVector(IPR_p));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "F_Q", getAverage2dVector(F_Q));
    totalError += computeError(pathObservables + "cell/" + ruleStr + "F_P", getAverage2dVector(F_P));
    totalError += computeError(pathObservables + "other/" + ruleStr + "B1", getAverage2dVector(B1));
    totalError += computeError(pathObservables + "other/" + ruleStr + "B2", getAverage2dVector(B2));
    totalError += computeError(pathObservables + "other/" + ruleStr + "B3", getAverage2dVector(B3));
    totalError += computeError(pathObservables + "other/" + ruleStr + "V1", getAverage2dVector(V1));
    totalError += computeError(pathObservables + "other/" + ruleStr + "V2", getAverage2dVector(V2));
    totalError += computeError(pathObservables + "other/" + ruleStr + "V3", getAverage2dVector(V3));
    return totalError;
}

double computeError(const std::string &observablePath, const std::vector<double> &simulatedObservable)
{
    std::vector<double> experimentalObservable{getValuesObservable(observablePath)};
    double numerator{0.};
    double denominator{0.};
    for (int i{0}; i < experimentalObservable.size(); ++i)
    {
        numerator += (experimentalObservable[i] - simulatedObservable[i]) *
                     (experimentalObservable[i] - simulatedObservable[i]);
        denominator += experimentalObservable[i] * experimentalObservable[i];
    }
    return numerator / denominator;
}

std::vector<double> getValuesObservable(const std::string &observablePath)
{
    std::vector<double> observable;
    std::ifstream file(observablePath + ".txt");
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            bool round{true};
            std::string number;
            for (const auto x : line)
            {
                if (round)
                {
                    if (x == ' ')
                    {
                        round = false;
                    }
                }
                else
                {
                    if (x == ' ')
                    {
                        observable.push_back(std::stod(number));
                        break;
                    }
                    else
                    {
                        number += x;
                    }
                }
            }
        }
    }
    else
    {
        std::cerr << "The file " << observablePath << " could not be opened/\n";
    }
    return observable;
}

void writeBestParameters(const std::string &filePath, const std::vector<double> &bestParameters)
{
    std::ofstream outFile(filePath, std::ios::app);
    if (outFile.is_open())
    {
        outFile << bestParameters[0];
        for (int iParameter{1}; iParameter < bestParameters.size(); ++iParameter)
        {
            outFile << " " << bestParameters[iParameter];
        }
        outFile << "\n";
    }
    else
    {
        std::cerr << "The file " << filePath << " could not be opened.\n";
    }
}

void oneMonteCarloStep(const MCMethod &method, const std::vector<bool> &parametersToChange, const int numberOfGames,
                       const std::string &pathObservables, std::vector<double> &parameters, double &averageError,
                       const int numberOfRounds, const int numberOfPlayers, const Rule &rule,
                       const std::vector<double> &parametersPlayersType, const std::vector<double> &parametersStarsCol,
                       const std::vector<double> &parametersStarsNeu, const std::vector<double> &parametersStarsDef)
{
    switch (method)
    {
    case MCMethod::one_pm:
    {
        // choice of iParameterToChange and epsilon
        int iParameterToChange;
        do
        {
            iParameterToChange = getRandomNumber(0, parameters.size() - 1);
        } while (!parametersToChange[iParameterToChange]);

        const double epsilon{getRandomSmallChange(parameters, iParameterToChange)};

        // + epsilon
        std::vector<double> parametersPlus{parameters};
        parametersPlus[iParameterToChange] += epsilon;
        double averageErrorPlus{getAverageError(numberOfRounds, numberOfPlayers, rule, parametersPlus,
                                                parametersPlayersType, parametersStarsCol, parametersStarsNeu,
                                                parametersStarsDef, numberOfGames, pathObservables)};
        if (averageErrorPlus < averageError)
        {
            parameters = parametersPlus;
            averageError = averageErrorPlus;
        }
        // - epsilon
        else
        {
            std::vector<double> parametersMinus{parameters};
            parametersMinus[iParameterToChange] -= epsilon;
            double averageErrorMinus{getAverageError(numberOfRounds, numberOfPlayers, rule, parametersMinus,
                                                     parametersPlayersType, parametersStarsCol, parametersStarsNeu,
                                                     parametersStarsDef, numberOfGames, pathObservables)};

            parameters = parametersMinus;
            averageError = averageErrorMinus;
        }
        break;
    }
    default:
        break;
    }
}

double getRandomSmallChange(const std::vector<double> &parameters, const int iParameterToChange)
{
    double epsilon{0.};
    switch (iParameterToChange)
    {
    case 0:
        do
        {
            epsilon = getRandomNumber(0., 1.) * 0.1;
        } while (parameters[0] - epsilon < 0 ||
                 parameters[0] + epsilon < 0 ||
                 parameters[0] - epsilon > 1 ||
                 parameters[0] + epsilon > 1);
        break;
    case 1:
        epsilon = getRandomNumber(0., 1.) * 0.2;
        break;
    case 2:
    case 4:
    case 6:
        epsilon = getRandomNumber(0., 1.) * 10;
        break;
    case 3:
    case 5:
    case 7:
        epsilon = getRandomNumber(0., 1.) * 0.2;
        break;
    default:
        std::cerr << "The parameter " << iParameterToChange << " is not implemented.\n";
        break;
    }
    return epsilon;
}
