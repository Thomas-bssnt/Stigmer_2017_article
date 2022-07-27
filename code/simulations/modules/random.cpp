#include <random>

#include "random.h"

double getRandomNumber()
{
    return std::uniform_real_distribution(0., 1.)(myRandom::engine);
}

double getRandomNumber(double min, double max)
{
    return std::uniform_real_distribution(min, max)(myRandom::engine);
}

int getRandomNumber(int min, int max)
{
    return std::uniform_int_distribution(min, max)(myRandom::engine);
}
