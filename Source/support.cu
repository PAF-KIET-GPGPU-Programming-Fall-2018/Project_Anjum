#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "support.h"


void verifyResults(int computed[N], int actual[N])
{
	for (int i = 0; i < N; i++)
	{
		if (computed[i] != actual[i])
			{
		   printf("TEST FAILED at index %d, actual = %d, computed = %d"
			   "\n\n", i, actual[i], computed[i]);
			exit(0);
			}
	}
	printf("TEST PASSED\n\n");
}

void printCSV(int(&values)[N][N])
{
	for (int i = 0; i<(N); i++)
	{
		for (int j = 0; j < (N); j++)
		{	
			std::cout << values[i][j] << ',';
		}
		std::cout << '\n';
	}

}

void printCSV(int(&values)[N])
{
	for (int i = 0; i<(N); i++)
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

	for (int i = 0; i < (N ); i++)

	{	for (int j = 0; j < (N ); j++)
		{values[i][j] = val[i*N + j];}
	}

}

void loadCSV(const std::string& filename, int(&values)[N])
{
	std::vector<float> val;
	
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



