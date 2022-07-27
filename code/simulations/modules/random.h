#ifndef RANDOM_H
#define RANDOM_H

#include <random>

namespace myRandom
{
    thread_local static std::random_device randomDevice;
    thread_local static std::mt19937 engine(randomDevice());
}

double getRandomNumber();

double getRandomNumber(double min, double max);

int getRandomNumber(int min, int max);

#endif
