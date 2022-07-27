#ifndef MC_FUNCTION_H
#define MC_FUNCTION_H

#include <string>
#include <vector>

#include "Game.h"

enum class MCMethod
{
    one_pm,
};

double getAverageError(const int numberOfRounds, const int numberOfPlayers, const Rule &rule,
                       const std::vector<double> &parametersVisits, const std::vector<double> &parametersPlayersType,
                       const std::vector<double> &parametersStarsCol, const std::vector<double> &parametersStarsNeu,
                       const std::vector<double> &parametersStarsDef, const int numberOfGames,
                       const std::string &pathDataFigures);

double computeError(const std::string &observablePath, const std::vector<double> &simulatedObservable);

std::vector<double> getValuesObservable(const std::string &observablePath);

void writeBestParameters(const std::string &filePath, const std::vector<double> &bestParameters);

void oneMonteCarloStep(const MCMethod &method, const std::vector<bool> &parametersToChange, const int numberOfGames,
                       const std::string &pathDataFigures, std::vector<double> &parameters, double &averageError,
                       const int numberOfRounds, const int numberOfPlayers, const Rule &rule,
                       const std::vector<double> &parametersPlayersType, const std::vector<double> &parametersStarsCol,
                       const std::vector<double> &parametersStarsNeu, const std::vector<double> &parametersStarsDef);

double getRandomSmallChange(const std::vector<double> &parameters, const int iParameterToChange);

#endif
