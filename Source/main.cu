#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <iostream> //for cout
#include <malloc.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "support.h"
#include "kernel.h"


#include "support.cu"
#include "kernel.cu"
#include "time.cu"


void onDevice(int matL_h[][N], int vecB_h[N], int vecX_h[N],int vecX_actual[N],int kernel)
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

	gpu_simple_solver(matL_d, vecX_d, vecB_d, N,kernel);
	cudaDeviceSynchronize();
	

	
	printf("Copying data from Device to Host...\n"); fflush(stdout);
	cuda_ret = cudaMemcpy(vecX_h, vecX_d, N*sizeof(int), cudaMemcpyDeviceToHost);
	if (cuda_ret != cudaSuccess) { std::cout << "Unable to Copy Data Back From Device \n"; }

cudaDeviceSynchronize();
	
		for (int i=0;i<N;i++)
	{
	printf("%d \n",vecX_h[i]);
	}
	
		verifyResults(vecX_h, vecX_actual);

		getchar();
	cudaFree(matL_d);
	cudaFree(vecX_d);
	cudaFree(vecB_d);


	
}
void onHost(int kernel)
{
  	std::stringstream ss;
	int vecX_actual[N];
	int	vecX_h[N];
	int vecB_h[N];
	int matL_h[N][N];
	int dataset = 1;
 
if (N>1023) { printf("Matrix of More then 1023 Width not Supported"); exit(0);}

if (N==8){ dataset = 0; }
else if (N==16){ dataset = 1; }
else if (N == 32) { dataset = 2; }
else { printf("Unknown Matrix Width"); exit(0); }

	memset(vecX_h, 0, N);

	ss.str("");
	ss << "dataset" << dataset << "/vecX.csv";
	loadCSV(ss.str(), vecX_actual);

	ss.str("");
	ss << "dataset" << dataset << "/vecB.csv";
	loadCSV(ss.str(), vecB_h);

	
	ss.str("");
	ss << "dataset" << dataset << "/matL.csv";
	loadCSV(ss.str(), matL_h);
		
	printCSV(vecB_h);
	printf("\n \n ");
	printCSV(matL_h);
	printf("\n \n ");
	
	onDevice(matL_h, vecB_h, vecX_h,vecX_actual,kernel);

	/*
	free(matL_h);
	free(vecB_h);
	free(vecX_actual);
	free(vecX_h);
	*/
}

//int main()

int main(int argc, char* argv[])
{ 	
int kernel = atoi(argv[1]);
onHost(kernel);
}
