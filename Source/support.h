#ifndef __FILEH__
#define __FILEH__

#include "dimention.h"




void loadCSV(const std::string& filename, int values[N][N]);
void loadCSV(const std::string& filename, int(&values)[N]);
void verifyResults(int computed[N], int actual[N]);
void writeCSV(std::string& filename, int(&vecXh)[N][N], int width, int totsize);

void printCSV(int(&values)[N][N]);
void printCSV(int(&values)[N]);



#endif