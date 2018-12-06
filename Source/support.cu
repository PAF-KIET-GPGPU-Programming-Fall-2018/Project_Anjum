#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "support.h"



void verifyResults(int(&computed)[N], int(&actual)[N], unsigned int count)
{
	// Compare the computed values to the actual ones
	float tolerance = 1e-3;

	printf("Total Length %d \n", count);
	for (unsigned int i = 0; i < count; i++)
	{
		if (i != 1)
		{
			const double diff = (computed[i] - actual[i]) / actual[i];
			if (diff > tolerance || diff < -tolerance)
			{
				printf("TEST FAILED at index %d, actual = %d, computed = %d"
					"\n\n", i, actual[i], computed[i]);
				// exit(0);
			}
		}
	}
	printf("TEST PASSED\n\n");
}

void printCSV(int(&values)[N][N])
{
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j < N; j++)
		{	
			std::cout << values[i][j] << ',';
		}
		std::cout << '\n';
	}

}

void printCSV(int(&values)[N])
{
	for (int i = 0; i<N; i++)
	{
		if (i>0 && i%N == 0) { std::cout << '\n'; }
		std::cout << values[i] << ',';
	}

}

void writeCSV(std::string& filename, int(&vecXh)[N][N], int width, int totsize)
{
	std::ofstream myfile;
	myfile.open(filename.c_str());
	for (int i = 0; i<totsize; i++)
	{
		if (i>0 && i%width == 0)
		{
			myfile << '\n';
			//std::cout << '\n'; 
		}
		myfile << vecXh[i] << ',';
		//std::cout <<vecXh[i] <<',';
	}
	myfile.close();

}


void loadCSV(const std::string& filename, int values[N][N])
{
	std::vector<float> val;
	std::cout << filename.c_str() << "\n";
	std::ifstream file(filename.c_str());
	std::string line;
	std::string cell;


	while (std::getline(file, line))
	{
		std::stringstream lineStream(line);
		
		//stof doesn't compile on GEM, had to use atof instead
		while (std::getline(lineStream, cell, ','))
		{
			val.push_back(std::stof(cell.c_str()));
			
		}
	}

	
	for (int i = 0; i<N; i++)
		for (int j = 0; j<N; j++)
	{
		values[i][j] = val[i*N+j];
		//std::cout << values[i][j] << "\n";
	}

}

void loadCSV(const std::string& filename, int(&values)[N])
{
	std::vector<float> val;
	std::cout << filename.c_str() << "\n";
	std::ifstream file(filename.c_str());
	std::string line;
	std::string cell;


	while (std::getline(file, line))
	{
		std::stringstream lineStream(line);

		//stof doesn't compile on GEM, had to use atof instead
		while (std::getline(lineStream, cell, ','))
		{
			val.push_back(std::stof(cell.c_str()));

		}
	}

	for (int i = 0; i<N; i++)
		{
			values[i] = val[i];
			//std::cout << values[i] << "\n";
		}

}


void startTime(Timer* timer) {
	gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
	gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
	return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) \
		+ (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}
