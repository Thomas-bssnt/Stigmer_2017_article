#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Game.h"
#include "Player.h"
#include "random.h"
#include "useful_functions.h"

std::string stringFromEnum(const Rule &rule)
{
    switch (rule)
    {
    case Rule::rule_1:
        return "rule_1/";
    case Rule::rule_2:
        return "rule_2/";
    case Rule::rule_3:
        return "rule_3/";
    case Rule::rule_4:
        return "rule_4/";
    }
}

std::string stringFromEnum(const PlayerType &playerType)
{
    switch (playerType)
    {
    case PlayerType::optimized_1:
        return "model_opt_1/";
    case PlayerType::optimized_2:
        return "model_opt_2/";
    default:
        return "";
    }
}

PlayerType getPlayerType(const std::vector<double> parameters)
{
    const double p{getRandomNumber()};
    if (p <= parameters[0])
    {
        return PlayerType::defector;
    }
    else if (p <= parameters[0] + parameters[1])
    {
        return PlayerType::neutral;
    }
    return PlayerType::collaborator;
}

int getScoreMax(const Rule &rule, const int numberOfRounds, const int Vmax1, const int Vmax2, const int Vmax3)
{
    switch (rule)
    {
    case Rule::rule_1:
    case Rule::rule_2:
        return numberOfRounds * (Vmax1 + Vmax2 + Vmax3);
    case Rule::rule_3:
    case Rule::rule_4:
        return numberOfRounds * (Vmax1 * 5 + Vmax2 * 3);
    }
}

std::vector<double> getParameters(const std::string &filePath)
{
    std::vector<double> bestParameters;
    std::ifstream outFile(filePath);
    if (outFile.is_open())
    {
        int iLine{-1};
        std::string lastLine;
        std::string nextLine;
        do
        {
            ++iLine;
            lastLine = nextLine;
        } while (std::getline(outFile, nextLine));

        if (iLine != 0) // if the file is not empty
        {
            int iNumber{0};
            std::string number;
            for (const auto x : lastLine + " ")
            {
                if (x == ' ')
                {
                    bestParameters.push_back(std::stod(number));
                    number = "";
                    ++iNumber;
                }
                else
                {
                    number += x;
                }
            }
        }
        else
        {
            std::cerr << "The file " << filePath << " is empty.\n";
        }
    }
    else
    {
        std::cerr << "The file " << filePath << " does not exist or could not be opened.\n";
    }
    return bestParameters;
}

double computeP(const std::vector<double> &starsDistribution, const std::vector<int> &values, const int Vmax)
{
    double P{0.};
    for (int iCell{0}; iCell < starsDistribution.size(); ++iCell)
    {
        P += starsDistribution[iCell] * values[iCell];
    }
    return P / Vmax;
}

double computeQ(const std::vector<double> &visitsDistribution, const std::vector<int> &values, const int Vmax1,
                const int Vmax2, const int Vmax3)
{
    double Q{0.};
    for (int iCell{0}; iCell < visitsDistribution.size(); ++iCell)
    {
        Q += visitsDistribution[iCell] * values[iCell];
    }
    return Q * 3. / (Vmax1 + Vmax2 + Vmax3);
}

double computeIPR(const std::vector<double> &distribution)
{
    double IPR{0.};
    for (const auto &value : distribution)
    {
        IPR += value * value;
    }
    return 1. / IPR;
}

double computeF(const std::vector<double> &distribution, const std::vector<int> &values)
{
    double sumSqrt{0.};
    int sumValues{0};
    for (int iCell{0}; iCell < distribution.size(); ++iCell)
    {
        sumValues += values[iCell];
        sumSqrt += std::sqrt(distribution[iCell] * values[iCell]);
    }
    return sumSqrt / std::sqrt(sumValues);
}

double getMedian(const std::vector<double> &vector)
{
    std::vector<double> sortedVector{vector};
    std::sort(sortedVector.begin(), sortedVector.end());
    if (sortedVector.size() % 2)
    {
        return sortedVector[sortedVector.size() / 2];
    }
    return (sortedVector[sortedVector.size() / 2] + sortedVector[sortedVector.size() / 2 - 1]) / 2;
}

double getAverage(const std::vector<double> &vector)
{

    return std::accumulate(vector.begin(), vector.end(), 0.) / vector.size();
}

std::vector<double> getAverage2dVector(const std::vector<std::vector<double>> &vector2d)
{
    std::vector<double> averagedVector(vector2d.size(), 0.);
    for (int i{0}; i < vector2d.size(); ++i)
    {
        averagedVector[i] = getAverage(vector2d[i]);
    }
    return averagedVector;
}

std::vector<double> getHistogram(const std::vector<double> &score, const int numberOfDivisions)
{
    // Do the histogram
    std::vector<int> counts(numberOfDivisions, 0);
    for (auto &value : score)
    {
        int binNumber{static_cast<int>(value < 1 ? value * numberOfDivisions : numberOfDivisions - 1)};
        counts[binNumber]++;
    }

    // Normalization (norm 1)
    std::vector<double> countsNormalized(numberOfDivisions, 0.);
    for (int iDivisions{0}; iDivisions < numberOfDivisions; ++iDivisions)
    {
        countsNormalized[iDivisions] = static_cast<double>(counts[iDivisions]) / score.size() * numberOfDivisions;
    }
    return countsNormalized;
}

std::vector<double> getL1Norm(std::vector<int> &vector)
{
    int sum{0};
    for (auto &value : vector)
    {
        sum += value;
    }
    std::vector<double> normalizedVector(vector.size(), 0);
    for (int i{0}; i < vector.size(); ++i)
    {
        normalizedVector[i] = vector[i] / static_cast<double>(sum);
    }
    return normalizedVector;
}

void getS_team_N_def(const std::vector<double> &teamScore, const std::vector<int> &numberDefectorGame,
                     std::vector<double> &mean, std::vector<double> &median)
{
    std::vector<int> numberGamesWithNdefectors(mean.size(), 0);
    for (int iGame{0}; iGame < teamScore.size(); ++iGame)
    {
        ++numberGamesWithNdefectors[numberDefectorGame[iGame]];
    }

    for (int iNumberDefector{0}; iNumberDefector < numberGamesWithNdefectors.size(); ++iNumberDefector)
    {
        std::vector<double> scores(numberGamesWithNdefectors[iNumberDefector], 0);
        int i{0};
        for (int iGame{0}; iGame < teamScore.size(); ++iGame)
        {
            if (numberDefectorGame[iGame] == iNumberDefector)
            {
                scores[i] = teamScore[iGame];
                ++i;
            }
        }
        if (numberGamesWithNdefectors[iNumberDefector] != 0)
        {
            mean[iNumberDefector] = getAverage(scores);
            median[iNumberDefector] = getMedian(scores);
        }
    }
}

std::vector<std::size_t> sortIndexes(const std::vector<double> &vector)
{
    std::vector<std::size_t> idx(vector.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&vector](std::size_t i1, std::size_t i2)
                     { return vector[i1] > vector[i2]; });
    return idx;
}

void saveObservable(const std::string &observablePath, const std::vector<double> &observable)
{
    std::ofstream file(observablePath + ".txt");
    if (file.is_open())
    {
        for (const auto &value : observable)
        {
            file << value << "\n";
        }
    }
    else
    {
        std::cerr << "The file " + observablePath + " could not be opened.\n";
    }
}
