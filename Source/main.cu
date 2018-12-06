#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <iostream> //for cout
#include <malloc.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "support.cu"
#include "kernel.cu"
#include "time.cu"
#include "support.h"
#include "kernel.h"
#include "time.h"


void onDevice(int matL_h[][N], int vecB_h[N], int vecX_actual[N])
{
	int* matL_d;
	int* vecX_d;
	int* vecB_d;

	cudaError_t cuda_ret;
	//	printf("Allocating device variables..."); fflush(stdout);
	cuda_ret = cudaMalloc((void **)&matL_d, N*N*sizeof(int));
	if (cuda_ret != cudaSuccess) { std::cout << "Unable to Allocate Memory"; }
	cuda_ret = cudaMalloc((void **)&vecX_d, N*sizeof(int));
	if (cuda_ret != cudaSuccess) { std::cout << "Unable to Allocate Memory"; }
	cuda_ret = cudaMalloc((void **)&vecB_d, N*sizeof(int));
	if (cuda_ret != cudaSuccess) { std::cout << "Unable to Allocate Memory"; }

	//	printf("Copying data from host to device...,%d", sizeof(matL_h)); fflush(stdout);
	cuda_ret = cudaMemcpy(matL_d, matL_h, N*N*sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_ret != cudaSuccess) { std::cout << "Unable to Copy Data in Memory"; }
	cuda_ret = cudaMemcpy(vecB_d, vecB_h, N*sizeof(int), cudaMemcpyHostToDevice);
	if (cuda_ret != cudaSuccess) { std::cout << "Unable to Copy Data in Memory"; }

	printf("\n \n Starting Calculation on Device \n");
	//gpu_Multiply <<<1, 8 >> >(matL_d,vecX_d,vecB_d);


	gpu_simple_solver(matL_d, vecX_d, vecB_d, N);
	cudaDeviceSynchronize();
	/*
	gpu_complex_solver(matL_d, vecX_d,  vecB_d, N);
	cudaDeviceSynchronize();
	*/

	//	printf("Copying data from Device to Host...\n"); fflush(stdout);
	cuda_ret = cudaMemcpy(vecX_actual, vecX_d, N*sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; i++)
	{
		printf("%d ", vecX_actual[i]);
	}

	cudaFree(matL_d);
	cudaFree(vecX_d);
	cudaFree(vecB_d);

	
}

int main()
{

	std::stringstream ss;
	int vecX_actual[N];
	int	vecX_h[N];
	int vecB_h[N];
	int matL_h[N][N];

/*
	int vecX_actual[N] = { 1,1,1,1,1,1,1,1 };
	int vecB_h[N] = { 1	,4,		9,		16	,	25,		36,		49,		64 };
	int matL_h[N][N] =
	{
		{ 1,0,0,0,0,0,0,0 },
		{ 2,2,0,0,0,0,0,0 },
		{ 3,3,3,0,0,0,0,0 },
		{ 4,4,4,4,0,0,0,0 },
		{ 5,5,5,5,5,0,0,0 },
		{ 6,6,6,6,6,6,0,0 },
		{ 7,7,7,7,7,7,7,0 },
		{ 8,8,8,8,8,8,8,8 }
	};
	
*/
if (N>1023) { printf("Matrix of More then 1023 Width not Supported"); exit(0);}

	memset(vecX_h, 0, N);
	ss.str("");
	ss << "vecX.csv";
	loadCSV(ss.str(), vecX_actual);

	ss.str("");
	ss << "vecB.csv";
	loadCSV(ss.str(), vecB_h);

	ss.str("");
	ss << "matL.csv";
	loadCSV(ss.str(), matL_h);

	
	onDevice(matL_h, vecB_h, vecX_actual);



}
