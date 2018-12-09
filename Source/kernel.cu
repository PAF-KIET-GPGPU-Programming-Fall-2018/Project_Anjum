#include <cuda_runtime.h>
#include "kernel.h"
#include "stdio.h"


#include <iostream> //for cout
#define L_Matrix(row,col) matL[((row)*numRows + (col))]
#define L_Matrix_mat(row,col) matrix[((row)*numRows + (col))]
#define L_Matrix_t(col,row) matL[((row)*numRows + (col))]

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)
//#define DEBUGGING 1



using namespace std;


__global__ void gpu_square_update_kernel_transposed(int* matL, int* vecX, int* vecB, int numRows)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	int y = idy * 2;
	int x = idx * 2;
	int top_tri_idx = y;

	if (x == 0)
	{
		vecB[y + 1] = (vecB[y + 1] - L_Matrix_t(top_tri_idx + 1, top_tri_idx) / L_Matrix_t(top_tri_idx, top_tri_idx)*vecB[y]) / L_Matrix_t(top_tri_idx + 1, top_tri_idx + 1);
		vecB[y] = vecB[y] / L_Matrix_t(top_tri_idx, top_tri_idx);
	}

	if (idx >= numRows / 2 || idy >= numRows / 2)
		return;

	if (idy <= idx)
		return;

	//element 1,0 (y,x) (row,col)
	L_Matrix_t(y + 1, x) = (L_Matrix_t(y + 1, x) - L_Matrix_t(top_tri_idx + 1, top_tri_idx) / L_Matrix_t(top_tri_idx, top_tri_idx)*L_Matrix_t(y, x)) / L_Matrix_t(top_tri_idx + 1, top_tri_idx + 1);

	//element 1,1 (y,x) (row,col)
	L_Matrix_t(y + 1, x + 1) = (L_Matrix_t(y + 1, x + 1) - L_Matrix_t(top_tri_idx + 1, top_tri_idx) / L_Matrix_t(top_tri_idx, top_tri_idx)*L_Matrix_t(y, x + 1)) / L_Matrix_t(top_tri_idx + 1, top_tri_idx + 1);

	//element 0,0 (y,x) (row,col)
	L_Matrix_t(y, x) = L_Matrix_t(y, x) / L_Matrix_t(top_tri_idx, top_tri_idx);

	//element 0,1 (y,x) (row,col)
	L_Matrix_t(y, x + 1) = L_Matrix_t(y, x + 1) / L_Matrix_t(top_tri_idx, top_tri_idx);
}

__global__ void gpu_Multiply(int* matL, int* vecX, int* vecB)
{
	//int idx = blockIdx.x*blockDim.x + threadIdx.x;


	for (int i = 0; i < blockDim.x; i++)
	{
		vecB[threadIdx.x] = vecB[threadIdx.x] + (vecX[threadIdx.x] * matL[(threadIdx.x * blockDim.x) + i]);
		//printf(" (%d, %d) \n",((threadIdx.x * blockDim.x)+i), matL[(threadIdx.x * blockDim.x)+i]);
	}


}

__global__ void gpu_simple_solver_kernel(int* matL, int* vecX, int* vecB, int numRows, int i)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= numRows)
		return;
	//update the B value for every thread by subtracting off the known x (which was calculating last iteration)
	//multiplied by the corresponding L element
	if (i != 0)
		vecB[idx] = vecB[idx] - matL[(idx*numRows + i) - 1] * vecX[i - 1];

	if (idx == i)
	{
		vecX[i] = vecB[i] / matL[i*numRows + i];
	}
}

__global__ void gpu_simple_solver_kernel2(int* matL, int* vecX, int* vecB, int numRows, int i)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= numRows)
		return;
	//update the B value for every thread by subtracting off the known x (which was calculating last iteration)
	//multiplied by the corresponding L element
	if (i != 0)
		vecB[idx] = vecB[idx] - matL[(idx*numRows + i) - 1] * vecX[i - 1];

	if (idx == i)
	{
		vecX[i] = vecB[i] / matL[i*numRows + i];
	}
}

__global__ void gpu_simple_solver_Anjum(int* matL, int* vecX, int* vecB, int numRows)
{
	int rs_B;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int ds_X[N];
	__shared__ int ds_matL[N*N];

	if (idx >= numRows)		return;
	rs_B = vecB[idx];
	ds_X[threadIdx.x] = vecX[idx];

	for (int i = 0; i < numRows; i++)
	{
		ds_matL[idx*numRows + i] = matL[(idx*numRows + i)];
	}
	__syncthreads();

	//update the B value for every thread by subtracting off the known x (which was calculating last iteration)
	//multiplied by the corresponding L element

	for (int j = 0; j < numRows; j++)
	{if (numRows != 0)
		{rs_B = rs_B - ds_matL[(idx*numRows + j) - 1] ;	}
		if (idx == j)
		{		ds_X[j] = rs_B / ds_matL[j*numRows + j];			}
	}

vecX[idx] = ds_X[idx];
}


__global__ void gpu_optimized_solver_Anjum(int* matL, int* vecX, int* vecB, int numRows)
{	int tot=0;
    int r_matL=0;
	__shared__ int ds_X[N];
 	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= numRows)		return;
  ds_X[idx]=0;

	for (int j = 0; j <numRows ; j++)
	{r_matL=matL[(j*numRows + idx) ];
	//__syncthreads();
      if (j> 0 && j>idx) 
		{
		tot= (-1 * (r_matL *ds_X[idx]));
		
		//atomicAdd (&ds_B[j],tot); //ds_B[j]+=tot; 	keepler takes time on shared memory for atomics then global memory
		atomicAdd (&vecB[j],tot); //vecB[idx]+=tot;		// Keepler Performs better on atomics on Global Memory then Shared memory
		}
		else if (idx == j)
		{
	  ds_X[j] = vecB[j] / r_matL	;	
		}
	}
vecX[idx] = ds_X[idx];

}



__global__ void gpu_initial_solver_Anjum(int* matL, int* vecX, int* vecB, int numRows)
{
	int rs_B;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row= blockIdx.y*blockDim.y + threadIdx.y;
	__shared__ int ds_X[N];
	__shared__ int ds_matL[N*N];

	if (col >= numRows || row >= numRows )		return;
	if (row >= col) 
 {
	rs_B = vecB[col];
	ds_X[col] = vecX[col];

	for (int i = 0; i < numRows; i++)
	{
		ds_matL[col*numRows + i] = matL[(col*numRows + i)];
	}
	__syncthreads();

	//update the B value for every thread by subtracting off the known x (which was calculating last iteration)
	//multiplied by the corresponding L element

	for (int j = 0; j < numRows; j++)
	{if (numRows != 0)
		{rs_B = rs_B - ds_matL[(col*numRows + j) - 1] ;	}
		if (col == j)
		{		ds_X[j] = rs_B / ds_matL[j*numRows + j];			}
	}

vecX[col] = ds_X[col];
 }
 
}

__global__ void gpu_square_solve_kernel_simple(int* matL, int* vecX, int* vecB, int numRows, int i)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int col_index = i * 2;
	if (col_index >= numRows)
		return;

	int row_index = idx;

	if (row_index < (i + 1) * 2 || row_index >= numRows)
		return;

	int value = matL[(row_index*numRows) + col_index] * vecB[col_index] + matL[(row_index*numRows) + col_index + 1] * vecB[col_index + 1];
	vecB[row_index] = vecB[row_index] - value;

}

void cpu_solver(int matL[N][N], int vecX[N], int vecB[N], int numRows)
{
    for (int i = 0; i < numRows; i++)
    {
        int val = vecB[i];
        for (int j = 0; j < i; j++)
        {
		  val = val - (matL[i][j]*vecX[j]);
		}
        vecX[i] = val / matL[i][i];
    }
}

void cpu_Multiply(int matL[N][N],int vecX[N],int vecB[N])
{
	for (int i = 0; i < (N-1); i++)
	{
		vecB[i] = 0;
		for (int j = 0; j < (N-1); j++)
		{

			if (matL[i][j] != 0) { vecB[i] = vecB[i] + (matL[i][j] * vecX[j]); }

		}
		printf("%d \n", vecB[i]);
	}
}

void gpu_complex_solver(int* matL, int* vecX, int* vecB, int numRows)
{
	dim3 dimGrid((numRows / 2 - 1) / 32 + 1, (numRows / 2 - 1) / 32 + 1, 1);
	dim3 dimBlock(32, 32, 1);

	//gpu_square_update_kernel << <dimGrid, dimBlock >> >(matL, vecX, vecB, numRows);

	const unsigned int numThreadsPerBlock = N;
	//const unsigned int numBlocks = (numRows - 1) / numThreadsPerBlock + 1;
	const unsigned int numBlocks = 1;
	// Executed 4 Times Only should 8 times for each Row
	for (int i = 0; i < (numRows / 2); i++)
	{
		gpu_square_solve_kernel_simple <<<numBlocks, numThreadsPerBlock >>>(matL, vecX, vecB, numRows, i);
	}

	//copy B to X for the verification code in main.cu
	cudaMemcpy(vecX, vecB, numRows * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
}

void gpu_simple_solver(int* matL, int* vecX, int* vecB, int numRows,int kernel)
{
	const unsigned int numThreadsPerBlock = N;
	//const unsigned int numBlocks = (numRows - 1) / numThreadsPerBlock + 1;
	const unsigned int numBlocks = 1;

if (kernel==1)
{ printf("\n Executing OLD gpu_simple_solver_kernel \n");
// Loop Below Executes 8 Times or for each Row of matL
	for (int i = 0; i < numRows; i++)
	{
		gpu_simple_solver_kernel <<<numBlocks, numThreadsPerBlock >>>(matL, vecX, vecB, numRows, i);
	}
}
else if (kernel==2)
{ 
	dim3 dimBlock(N,N,1);
	printf("\n Executing Initial gpu_initial_solver_Anjum \n");
	gpu_initial_solver_Anjum<<<numBlocks, dimBlock >>>(matL, vecX, vecB, numRows);
}
else if (kernel==3) 
{
printf("\n Executing gpu_simple_solver_kernel2 \n");
for (int i = 0; i < numRows; i++)
	{
		gpu_simple_solver_kernel2 <<<numBlocks, numThreadsPerBlock >>>(matL, vecX, vecB, numRows, i);
	}
}
else if (kernel==4) 
{
printf("\n Executing gpu_simple_solver_Anjum \n");
gpu_simple_solver_Anjum <<<numBlocks, numThreadsPerBlock >>>(matL, vecX, vecB, numRows);

}
else if (kernel==5) 
{
printf("\n Executing gpu_optimized_solver_Anjum \n");
gpu_optimized_solver_Anjum <<<numBlocks, numThreadsPerBlock >>>(matL, vecX, vecB, numRows);
}
else { 
printf("Unknown Kernel"); exit(0); 
}


	
}
