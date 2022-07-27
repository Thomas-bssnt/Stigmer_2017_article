#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "Game.h"

Game::Game(const int numberOfRounds, const int numberOfTurns, const int numberOfPlayers, const int numberOfCells,
           const int numberOfStarsPerCells, const Rule &rule, const MapType &mapType, const double tauEvaporation)
    : m_numberOfRounds{numberOfRounds},
      m_numberOfTurns{numberOfTurns},
      m_numberOfPlayers{numberOfPlayers},
      m_numberOfCells{numberOfCells},
      m_numberOfStarsPerCells{numberOfStarsPerCells},
      m_rule{rule},
      m_mapType{mapType},
      m_tauEvaporation{tauEvaporation},
      m_maxNumberOfStarsPerPlayerInARound{(m_rule == Rule::rule_1 || m_rule == Rule::rule_2)
                                              ? 5 * m_numberOfTurns
                                              : 8},
      m_map{creationOfTheMap()},
      m_vMap{std::vector<int>(m_numberOfCells, 0)},
      m_sMap{std::vector<int>(m_numberOfCells, 0)},
      m_sColorsMap{std::vector<double>(m_numberOfCells, 0.)},
      m_colorsDistribution{std::vector<double>(m_numberOfCells, 1. / m_numberOfCells)},
      m_instantaneousVisitsDistribution{std::vector<double>(m_numberOfCells, 1. / m_numberOfCells)},
      m_instantaneousStarsDistribution{std::vector<double>(m_numberOfCells, 1. / m_numberOfCells)},
      m_visitsDistribution{std::vector<double>(m_numberOfCells, 1. / m_numberOfCells)},
      m_starsDistribution{std::vector<double>(m_numberOfCells, 1. / m_numberOfCells)},
      m_scores{std::vector<int>(m_numberOfPlayers, 0)},
      m_iCurrentRound{0},
      m_iTurn{std::vector<int>(numberOfPlayers, 0)},
      m_iCellOpened{std::vector<std::vector<std::vector<int>>>(m_numberOfPlayers,
                                                               std::vector<std::vector<int>>(m_numberOfRounds,
                                                                                             std::vector<int>(
                                                                                                 numberOfTurns,
                                                                                                 -1)))},
      m_sCellOpened{std::vector<std::vector<std::vector<int>>>(m_numberOfPlayers,
                                                               std::vector<std::vector<int>>(m_numberOfRounds,
                                                                                             std::vector<int>(
                                                                                                 numberOfTurns,
                                                                                                 0)))}
{
    assert(m_rule == Rule::rule_1 ||
           m_rule == Rule::rule_2 ||
           m_rule == Rule::rule_3 ||
           m_rule == Rule::rule_4);
    assert(m_mapType == MapType::random);
    assert(m_numberOfCells == 225);
}

Game::Game(const int numberOfRounds, const int numberOfPlayers, const Rule rule)
    : Game(numberOfRounds, 3, numberOfPlayers, 225, 5, rule, MapType::random, 1e6)
{
}

int Game::openACell(int iPlayer, int iCell)
{
    if (m_iCurrentRound < m_numberOfRounds)
    {
        if (m_iTurn[iPlayer] < m_numberOfTurns)
        {
            if (m_iCellOpened[iPlayer][m_iCurrentRound][m_iTurn[iPlayer]] == -1)
            {
                if (0 <= iCell && iCell < m_numberOfCells)
                {
                    if (!hasThePlayerOpenedTheCellDuringTheRound(iPlayer, iCell))
                    {
                        m_iCellOpened[iPlayer][m_iCurrentRound][m_iTurn[iPlayer]] = iCell;
                        return m_map[iCell];
                    }
                    else
                    {
                        std::cerr << "The player " << iPlayer << " already opened the cell " << iCell
                                  << " during the round.\n";
                        return -1; // change that
                    }
                }
                else
                {
                    std::cerr << "The cell number " << iCell << " does not exist.\n";
                    return -1; // change that
                }
            }
            else
            {
                std::cerr << "The player " << iPlayer << " already opened a cell. He needs to put stars now.\n";
                return -1; // change that
            }
        }
        else
        {
            std::cerr << "The player " << iPlayer << " already played " << m_numberOfTurns
                      << " times during the round.\n";
            return -1; // change that
        }
    }
    else
    {
        std::cerr << "The game is over.\n";
        return -1; // change that
    }
}

void Game::putStars(int iPlayer, int numberOfStars)
{
    if (m_iTurn[iPlayer] < m_numberOfTurns)
    {
        if (m_iCellOpened[iPlayer][m_iCurrentRound][m_iTurn[iPlayer]] != -1)
        {
            if (0 <= numberOfStars && numberOfStars <= m_numberOfStarsPerCells)
            {
                const int numberOfStarsRemaining{std::accumulate(m_sCellOpened[iPlayer][m_iCurrentRound].begin(),
                                                                 m_sCellOpened[iPlayer][m_iCurrentRound].end(),
                                                                 m_maxNumberOfStarsPerPlayerInARound,
                                                                 std::minus<>())};
                if (numberOfStars <= numberOfStarsRemaining)
                {
                    m_sCellOpened[iPlayer][m_iCurrentRound][m_iTurn[iPlayer]] = numberOfStars;
                    ++m_iTurn[iPlayer];

                    // change round if necessary
                    if (std::all_of(m_iTurn.begin(),
                                    m_iTurn.end(),
                                    [&numberOfTurns = m_numberOfTurns](int iTurn)
                                    { return iTurn == numberOfTurns; }))
                    {
                        changeRound();
                    }
                }
                else
                {
                    std::cerr << "There are only " << numberOfStarsRemaining << " stars remaining.\n";
                }
            }
            else
            {
                std::cerr << "The player " << iPlayer << " entered an invalid number of stars.\n";
            }
        }
        else
        {
            std::cerr << "The player " << iPlayer << " needs to choose a cell before putting stars.\n";
        }
    }
    else
    {
        std::cerr << "The player " << iPlayer << " already played " << m_numberOfTurns << " times during the round.\n";
    }
}

Game *Game::getAddress()
{
    return this;
}

const std::vector<int> &Game::getValues() const
{
    return m_map;
}

const std::vector<double> &Game::getColors() const
{
    return m_colorsDistribution;
}

const std::vector<double> &Game::getVisitsDistribution() const
{
    return m_visitsDistribution;
}

const std::vector<double> &Game::getInstantaneousVisitsDistribution() const
{
    return m_instantaneousVisitsDistribution;
}

const std::vector<double> &Game::getStarsDistribution() const
{
    return m_starsDistribution;
}

const std::vector<double> &Game::getInstantaneousStarsDistribution() const
{
    return m_instantaneousStarsDistribution;
}

const int &Game::getScoreOfPlayer(int iPlayer) const
{
    return m_scores[iPlayer];
}

const int &Game::getNumberOfRounds() const
{
    return m_numberOfRounds;
}

const int &Game::getNumberOfTurns() const
{
    return m_numberOfTurns;
}

const int &Game::getNumberOfCells() const
{
    return m_numberOfCells;
}

const Rule &Game::getRuleNumber() const
{
    return m_rule;
}

const int &Game::getCurrentRound() const
{
    return m_iCurrentRound;
}

void Game::changeRound()
{
    // Update the instantaneous distributions
    std::vector<int> vMapInTheCurrentRound(m_numberOfCells, 0);
    std::vector<int> sMapInTheCurrentRound(m_numberOfCells, 0);
    for (int iPlayer{0}; iPlayer < m_numberOfPlayers; ++iPlayer)
    {
        for (int iTurn{0}; iTurn < m_numberOfTurns; ++iTurn)
        {
            ++vMapInTheCurrentRound[m_iCellOpened[iPlayer][m_iCurrentRound][iTurn]];
            sMapInTheCurrentRound[m_iCellOpened[iPlayer][m_iCurrentRound][iTurn]] += m_sCellOpened[iPlayer][m_iCurrentRound][iTurn];
        }
    }
    m_instantaneousVisitsDistribution = getNorm1(vMapInTheCurrentRound);
    m_instantaneousStarsDistribution = getNorm1(sMapInTheCurrentRound);

    // Update of the cumulated distributions & colors
    const double evaporationFactor{m_tauEvaporation <= 1000. ? 1. - 1. / m_tauEvaporation : 1.};
    for (int iCell{0}; iCell < m_numberOfCells; ++iCell)
    {
        m_vMap[iCell] += vMapInTheCurrentRound[iCell];
        m_sMap[iCell] += sMapInTheCurrentRound[iCell];
        m_sColorsMap[iCell] = m_sColorsMap[iCell] * evaporationFactor + sMapInTheCurrentRound[iCell];
    }
    m_visitsDistribution = getNorm1(m_vMap);
    m_starsDistribution = getNorm1(m_sMap);
    m_colorsDistribution = getNorm1(m_sColorsMap);

    // update scores
    updateScores();

    // increase the round number
    ++m_iCurrentRound;
    m_iTurn = std::vector<int>(m_numberOfPlayers, 0);
}

void Game::updateScores()
{
    for (int iPlayer{0}; iPlayer < m_numberOfPlayers; ++iPlayer)
    {
        switch (m_rule)
        {
        case Rule::rule_1:
        case Rule::rule_2:
            for (int iTurn{0}; iTurn < m_numberOfTurns; ++iTurn)
            {
                m_scores[iPlayer] += m_map[m_iCellOpened[iPlayer][m_iCurrentRound][iTurn]];
            }
            break;
        case Rule::rule_3:
            for (int iTurn{0}; iTurn < m_numberOfTurns; ++iTurn)
            {
                m_scores[iPlayer] += m_map[m_iCellOpened[iPlayer][m_iCurrentRound][iTurn]] *
                                     m_sCellOpened[iPlayer][m_iCurrentRound][iTurn];
            }
            break;
        case Rule::rule_4:
            for (int iTurn{0}; iTurn < m_numberOfTurns; ++iTurn)
            {
                m_scores[iPlayer] += m_map[m_iCellOpened[iPlayer][m_iCurrentRound][iTurn]] *
                                     m_sCellOpened[iPlayer][m_iCurrentRound][iTurn];
            }
            m_scores[iPlayer] += std::accumulate(m_sCellOpened[iPlayer][m_iCurrentRound].begin(),
                                                 m_sCellOpened[iPlayer][m_iCurrentRound].end(),
                                                 m_maxNumberOfStarsPerPlayerInARound,
                                                 std::minus<>()) *
                                 50;
            break;
        }
    }
}

std::vector<int> Game::creationOfTheMap() const
{
    switch (m_mapType)
    {
    case MapType::random:
        if (m_numberOfCells == 225)
        {
            return {3, 1, 2, 27, 51, 2, 2, 3, 2, 0, 2, 0, 2, 3, 6,
                    2, 2, 1, 8, 2, 1, 0, 12, 1, 2, 3, 2, 1, 1, 1,
                    3, 2, 4, 2, 7, 0, 0, 12, 0, 71, 1, 3, 53, 1, 9,
                    72, 2, 2, 2, 6, 8, 19, 3, 1, 1, 72, 1, 1, 1, 14,
                    1, 1, 7, 0, 43, 7, 1, 0, 4, 2, 1, 1, 53, 3, 0,
                    86, 3, 2, 6, 1, 45, 20, 7, 24, 2, 4, 27, 3, 2, 3,
                    5, 3, 84, 3, 2, 0, 0, 3, 2, 27, 3, 1, 3, 14, 2,
                    3, 4, 13, 1, 3, 2, 1, 1, 24, 6, 53, 3, 3, 2, 19,
                    1, 1, 20, 4, 1, 2, 13, 21, 22, 0, 2, 2, 0, 1, 6,
                    2, 2, 13, 1, 0, 0, 24, 86, 0, 3, 2, 2, 1, 3, 20,
                    0, 1, 46, 85, 0, 2, 43, 3, 1, 1, 1, 1, 6, 5, 2,
                    3, 6, 2, 1, 6, 2, 0, 71, 3, 8, 2, 3, 4, 20, 20,
                    8, 2, 3, 4, 2, 2, 1, 1, 2, 0, 1, 3, 2, 45, 0,
                    2, 4, 44, 2, 1, 1, 3, 4, 2, 7, 4, 2, 3, 44, 3,
                    44, 99, 0, 28, 3, 0, 4, 2, 1, 9, 1, 0, 8, 11, 1};
        }
        else
        {
            std::cerr << "The map could not be generated.\n";
            return std::vector<int>(m_numberOfCells, -1);
        }
    default:
        std::cerr << "The map could not be generated.\n";
        return std::vector<int>(m_numberOfCells, -1);
    }
}

bool Game::hasThePlayerOpenedTheCellDuringTheRound(const int iPlayer, const int iCell) const
{
    return std::any_of(m_iCellOpened[iPlayer][m_iCurrentRound].begin(),
                       m_iCellOpened[iPlayer][m_iCurrentRound].begin() + m_iTurn[iPlayer] + 1,
                       [iCell](int iCellOpened)
                       { return iCellOpened == iCell; });
}

template <typename T>
std::vector<double> Game::getNorm1(const std::vector<T> &vector)
{
    /* all elements of the vector must be non-negative */
    const double sum{std::accumulate(vector.begin(), vector.end(), 0.)};
    if (sum != 0)
    {
        std::vector<double> normalizedVector(vector.begin(), vector.end());
        for (auto &value : normalizedVector)
        {
            value /= sum;
        }
        return normalizedVector;
    }
    else
    {
        return std::vector<double>(vector.size(), 1. / static_cast<double>(vector.size()));
    }
}
