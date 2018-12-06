#ifndef __FILEH__
#define __FILEH__

// include <sys/time.h>

#include "time.h"

#define N 8




typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void loadCSV(const std::string& filename, int values[N][N]);
void loadCSV(const std::string& filename, int(&values)[N]);
void verifyResults(int(&computed)[N], int(&actual)[N], unsigned int count);
void writeCSV(std::string& filename, int(&vecXh)[N][N], int width, int totsize);

void printCSV(int(&values)[N][N]);
void printCSV(int(&values)[N]);

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);




#endif