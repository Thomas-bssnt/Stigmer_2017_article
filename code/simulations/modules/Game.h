#ifndef GAME_H
#define GAME_H

#include <vector>

enum class Rule
{
    rule_1,
    rule_2,
    rule_3,
    rule_4,
};

enum class MapType
{
    random,
    continuous_1,
    continuous_2,
};

class Game
{
public:
    Game(int numberOfRounds, int numberOfTurns, int numberOfPlayers, int numberOfCells, int numberOfStarsPerTurn,
         const Rule &rule, const MapType &mapType, double tauEvaporation);

    Game(int numberOfRounds, int numberOfPlayers, Rule rule);

    [[nodiscard]] int openACell(int playerId, int iCell);

    void putStars(int playerId, int numberOfStars);

    Game *getAddress();

    [[nodiscard]] const std::vector<int> &getValues() const;

    [[nodiscard]] const std::vector<double> &getColors() const;

    [[nodiscard]] const std::vector<double> &getVisitsDistribution() const;

    [[nodiscard]] const std::vector<double> &getInstantaneousVisitsDistribution() const;

    [[nodiscard]] const std::vector<double> &getStarsDistribution() const;

    [[nodiscard]] const std::vector<double> &getInstantaneousStarsDistribution() const;

    [[nodiscard]] const int &getScoreOfPlayer(int playerId) const;

    [[nodiscard]] const int &getNumberOfRounds() const;

    [[nodiscard]] const int &getNumberOfTurns() const;

    [[nodiscard]] const int &getNumberOfCells() const;

    [[nodiscard]] const Rule &getRuleNumber() const;

    [[nodiscard]] const int &getCurrentRound() const;

private:
    void changeRound();

    void updateScores();

    [[nodiscard]] std::vector<int> creationOfTheMap() const;

    [[nodiscard]] bool hasThePlayerOpenedTheCellDuringTheRound(int playerId, int iCell) const;

    template <typename T>
    [[nodiscard]] static std::vector<double> getNorm1(std::vector<T> const &vector);

    // Global variables
    const int m_numberOfRounds;
    const int m_numberOfTurns;
    const int m_numberOfPlayers;
    const int m_numberOfCells;
    const int m_numberOfStarsPerCells;
    const Rule m_rule;
    const MapType m_mapType;
    const double m_tauEvaporation;
    const int m_maxNumberOfStarsPerPlayerInARound;
    const std::vector<int> m_map;
    //
    std::vector<int> m_vMap;
    std::vector<int> m_sMap;
    std::vector<double> m_sColorsMap;
    // Distributions
    std::vector<double> m_colorsDistribution;
    std::vector<double> m_instantaneousVisitsDistribution;
    std::vector<double> m_instantaneousStarsDistribution;
    std::vector<double> m_visitsDistribution;
    std::vector<double> m_starsDistribution;
    // Scores
    std::vector<int> m_scores;
    // Game variables
    int m_iCurrentRound;
    std::vector<int> m_iTurn;
    std::vector<std::vector<std::vector<int>>> m_iCellOpened;
    std::vector<std::vector<std::vector<int>>> m_sCellOpened;
};

#endif
