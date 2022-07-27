#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "Game.h"
#include "MC_functions.h"
#include "useful_functions.h"

int main()
{
    // Parameters of the program
    const int numberOfGamesInEachStep{300000};
    const MCMethod method{MCMethod::one_pm};
    const std::vector<bool> parametersToChange{true, true, false, false, false, false, false, false};

    // Parameters of the game
    const int numberOfRounds{20};
    const int numberOfPlayers{5};
    const Rule rule{Rule::rule_1};

    // Path of the in and out files
    const std::string pathDataFigures{"../../../data_figures/"};
    const std::string ruleStr{stringFromEnum(rule)};
    const std::string pathParameters{pathDataFigures + "model/parameters/"};
    const std::string pathObservables{pathDataFigures + "exp/"};

    // Get parameters
    std::vector<double> bestParametersVisits{getParameters(pathParameters + ruleStr + "cells.txt")};
    const std::vector<double> parametersPlayersType{getParameters(pathParameters + ruleStr + "players_type.txt")};
    const std::vector<double> parametersStarsCol{getParameters(pathParameters + "stars_col.txt")};
    const std::vector<double> parametersStarsNeu{getParameters(pathParameters + "stars_neu.txt")};
    const std::vector<double> parametersStarsDef{getParameters(pathParameters + "stars_def.txt")};

    // Get best parameters and best error
    double bestAverageError{getAverageError(numberOfRounds, numberOfPlayers, rule, bestParametersVisits,
                                            parametersPlayersType, parametersStarsCol, parametersStarsNeu,
                                            parametersStarsDef, numberOfGamesInEachStep, pathObservables)};
    std::cerr << "err =  " << bestAverageError << "\n\n";

    // The MC simulation
    while (true)
    {
        std::time_t t0, t1;
        time(&t0);

        std::vector<double> parameters{bestParametersVisits};
        double averageError{bestAverageError};
        oneMonteCarloStep(method, parametersToChange, numberOfGamesInEachStep, pathObservables, parameters,
                          averageError, numberOfRounds, numberOfPlayers, rule, parametersPlayersType,
                          parametersStarsCol, parametersStarsNeu, parametersStarsDef);
        if (averageError < bestAverageError)
        {
            bestAverageError = averageError;
            bestParametersVisits = parameters;
        }
        writeBestParameters(pathParameters + ruleStr + "cells.txt", bestParametersVisits);

        time(&t1);
        std::cerr << "t = " << std::difftime(t1, t0) << "s, "
                  << "err =  " << averageError << "\n\n";
    }

    return 0;
}
