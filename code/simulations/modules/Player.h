#ifndef PLAYER_H
#define PLAYER_H

#include <vector>

#include "Game.h"

struct Cell
{
    int index;
    int value;
};

enum class PlayerType
{
    collaborator,
    neutral,
    defector,
    optimized_1,
    optimized_2,
};

class Player
{
public:
    Player(int iPlayer, Game *pGame, const std::vector<double> &parametersVisits,
           const std::vector<double> &parametersStars, const PlayerType &playerType);

    void playARound();

    [[nodiscard]] const std::vector<std::vector<int>> &getPlayBestCellsRound() const;

    [[nodiscard]] const std::vector<std::vector<int>> &getValueBestCells() const;

    [[nodiscard]] const PlayerType &getPlayerType() const;

private:
    std::vector<double> computeExploringProbabilities();

    int choseACell(const std::vector<Cell> &cellsPlayed, const std::vector<double> &exploringCumProbabilities);

    static bool replayCell(int value, double criticalValue, double slope);

    int chooseACellByExploring(const std::vector<double> &exploringCumProbabilities,
                               const std::vector<Cell> &cellsPlayed);

    int choseNumberOfStars(int value);

    void updateBestCells(const std::vector<Cell> &cellsPlayed);

    static std::vector<double> getCumulativeSum(const std::vector<double> &vector);

    static bool isSelectedCellIn(int iSelectedCell, const std::vector<Cell> &cells);

    // Player variables
    const int m_iPlayer;
    // Game variables
    Game *mp_Game;
    const Rule m_ruleNumber;
    const int m_numberOfTurns;
    const int m_numberOfCells;
    int m_round;
    // Strategies variables
    const std::vector<double> m_parametersVisits;
    const std::vector<double> m_parametersStars;
    const PlayerType m_playerType;
    std::vector<Cell> m_bestCells;
    std::vector<std::vector<int>> m_playBestCellsRound;
    std::vector<std::vector<int>> m_valueBestCells;
};

#endif
