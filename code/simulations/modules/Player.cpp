#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "Player.h"
#include "random.h"

Player::Player(const int iPlayer, Game *pGame, const std::vector<double> &parametersVisits,
               const std::vector<double> &parametersStars, const PlayerType &playerType)
    : m_iPlayer{iPlayer},
      mp_Game{pGame},
      m_ruleNumber{mp_Game->getRuleNumber()},
      m_numberOfTurns{mp_Game->getNumberOfTurns()},
      m_numberOfCells{mp_Game->getNumberOfCells()},
      m_round{0},
      m_parametersVisits{parametersVisits},
      m_parametersStars{parametersStars},
      m_playerType{playerType},
      m_bestCells{std::vector<Cell>(m_numberOfTurns, {-1, -1})},
      m_playBestCellsRound{std::vector<std::vector<int>>(m_numberOfTurns,
                                                         std::vector<int>(mp_Game->getNumberOfRounds(), 0))},
      m_valueBestCells{std::vector<std::vector<int>>(m_numberOfTurns,
                                                     std::vector<int>(mp_Game->getNumberOfRounds(), 0))}
{
    assert(m_ruleNumber == Rule::rule_1 ||
           m_ruleNumber == Rule::rule_2);
    assert(m_numberOfTurns == 3);
}

void Player::playARound()
{
    m_round = mp_Game->getCurrentRound();
    const std::vector<double> exploringProbabilities{computeExploringProbabilities()};
    std::vector<Cell> cellsPlayed;
    for (int iTurn{0}; iTurn < m_numberOfTurns; ++iTurn)
    {
        int iCell = choseACell(cellsPlayed, exploringProbabilities);
        int vCell = mp_Game->openACell(m_iPlayer, iCell);
        mp_Game->putStars(m_iPlayer, choseNumberOfStars(vCell));
        cellsPlayed.push_back({iCell, vCell});
    }
    updateBestCells(cellsPlayed);

    for (int iTurn{0}; iTurn < m_numberOfTurns; ++iTurn)
    {
        m_valueBestCells[iTurn][m_round] = m_bestCells[iTurn].value;
    }
}

std::vector<double> Player::computeExploringProbabilities()
{
    const std::vector<double> colors{mp_Game->getColors()};
    std::vector<double> powers(m_numberOfCells, 0.);
    double sumPowers{0.};
    for (int iCell{0}; iCell < m_numberOfCells; ++iCell)
    {
        powers[iCell] = std::pow(colors[iCell], m_parametersVisits[1]);
        sumPowers += powers[iCell];
    }
    std::vector<double> exploringProbabilities(m_numberOfCells, 0.);
    for (int iCell{0}; iCell < m_numberOfCells; ++iCell)
    {
        exploringProbabilities[iCell] = m_parametersVisits[0] / m_numberOfCells +
                                        (1 - m_parametersVisits[0]) * powers[iCell] / sumPowers;
    }
    return exploringProbabilities;
}

int Player::choseACell(const std::vector<Cell> &cellsPlayed, const std::vector<double> &exploringProbabilities)
{
    switch (cellsPlayed.size())
    {
    case 0:
        if (replayCell(m_bestCells[2].value, m_parametersVisits[2], m_parametersVisits[3]))
        {
            ++m_playBestCellsRound[2][m_round];
            return m_bestCells[2].index;
        }
        else
        {
            return chooseACellByExploring(exploringProbabilities, cellsPlayed);
        }
    case 1:
        if (replayCell(m_bestCells[1].value, m_parametersVisits[4], m_parametersVisits[5]))
        {
            ++m_playBestCellsRound[1][m_round];
            return m_bestCells[1].index;
        }
        else
        {
            return chooseACellByExploring(exploringProbabilities, cellsPlayed);
        }
    case 2:
        if (replayCell(m_bestCells[0].value, m_parametersVisits[6], m_parametersVisits[7]))
        {
            ++m_playBestCellsRound[0][m_round];
            return m_bestCells[0].index;
        }
        else
        {
            return chooseACellByExploring(exploringProbabilities, cellsPlayed);
        }
    default:
        std::cerr << "The player had already played 3 times.\n";
        return -1;
    }
}

bool Player::replayCell(const int value, const double criticalValue, const double slope)
{
    if (value < 0 ||
        value < criticalValue)
    {
        return false;
    }
    return getRandomNumber() < slope * (value - criticalValue) / 99.;
}

int Player::chooseACellByExploring(const std::vector<double> &exploringProbabilities,
                                   const std::vector<Cell> &cellsPlayed)
{
    double sum_best{0.};
    if (m_round > 0)
    {
        for (auto &cell : m_bestCells)
        {
            sum_best += exploringProbabilities[cell.index];
        }
    }
    double sum_exploration{0.};
    for (auto &cell : cellsPlayed)
    {
        if (!isSelectedCellIn(cell.index, m_bestCells))
        {
            sum_exploration += exploringProbabilities[cell.index];
        }
    }
    const double p{getRandomNumber() * (1. - sum_best - sum_exploration)};

    int iSelectedCell{-1};
    double proba{0.};
    while (proba < p)
    {
        ++iSelectedCell;
        if (!isSelectedCellIn(iSelectedCell, m_bestCells) &&
            !isSelectedCellIn(iSelectedCell, cellsPlayed))
        {
            proba += exploringProbabilities[iSelectedCell];
        }
    }
    return iSelectedCell;
}

int Player::choseNumberOfStars(const int vCell)
{
    double p0{0.};
    double p5{0.};
    switch (m_playerType)
    {
    case PlayerType::collaborator:
    case PlayerType::defector:
    case PlayerType::optimized_1:
    case PlayerType::optimized_2:
        p0 = (1. + std::tanh(m_parametersStars[1] * (vCell - m_parametersStars[0]) / 99.)) / 2.;
        p5 = (1. + std::tanh(m_parametersStars[3] * (vCell - m_parametersStars[2]) / 99.)) / 2.;
        break;
    case PlayerType::neutral:
        p0 = m_parametersStars[0];
        p5 = m_parametersStars[1];
        break;
    }
    const double p1234{(1. - p0 - p5) / 4.};

    // Select the number of stars
    std::vector<double> cumulatedStarDistribution{
        getCumulativeSum(std::vector<double>{p0, p1234, p1234, p1234, p1234, p5})};
    int iStar{0};
    const double p{getRandomNumber()};
    while (cumulatedStarDistribution[iStar] < p)
    {
        ++iStar;
    }
    return iStar;
}

void Player::updateBestCells(const std::vector<Cell> &cellsPlayed)
{
    m_bestCells = std::vector<Cell>(m_numberOfTurns, {-1, -1});
    for (auto &cellPlayed : cellsPlayed)
    {
        if (cellPlayed.value > m_bestCells[0].value)
        {
            m_bestCells[2] = m_bestCells[1];
            m_bestCells[1] = m_bestCells[0];
            m_bestCells[0] = cellPlayed;
        }
        else if (cellPlayed.value <= m_bestCells[0].value &&
                 cellPlayed.value > m_bestCells[1].value &&
                 cellPlayed.index != m_bestCells[0].index)
        {
            m_bestCells[2] = m_bestCells[1];
            m_bestCells[1] = cellPlayed;
        }
        else if (cellPlayed.value <= m_bestCells[1].value &&
                 cellPlayed.value > m_bestCells[2].value &&
                 cellPlayed.index != m_bestCells[1].index)
        {
            m_bestCells[2] = cellPlayed;
        }
    }
}

const std::vector<std::vector<int>> &Player::getPlayBestCellsRound() const
{
    return m_playBestCellsRound;
}

const std::vector<std::vector<int>> &Player::getValueBestCells() const
{
    return m_valueBestCells;
}

const PlayerType &Player::getPlayerType() const
{
    return m_playerType;
}

std::vector<double> Player::getCumulativeSum(const std::vector<double> &vector)
{
    std::vector<double> cumulatedVector(vector.size());
    std::partial_sum(vector.begin(), vector.end(), cumulatedVector.begin());
    return cumulatedVector;
}

bool Player::isSelectedCellIn(const int iSelectedCell, const std::vector<Cell> &cells)
{
    return std::any_of(cells.begin(), cells.end(), [iSelectedCell](Cell cell)
                       { return cell.index == iSelectedCell; });
}
