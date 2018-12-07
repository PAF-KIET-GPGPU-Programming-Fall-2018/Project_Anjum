%%cu 
#include <stdio.h>
#include <stdlib.h>
#include <iostream> //for cout
#include <malloc.h>
#include <cuda_runtime.h>
#define L_Matrix_mat(row,col) matrix[((row)*numRows + (col))]
#define L_Matrix_t(col,row) matL[((row)*numRows + (col))]
#define N 8


 
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
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	
	for (int i = 0; i < blockDim.x; i++)
	{
		vecB[threadIdx.x] = vecB[threadIdx.x] +(vecX[threadIdx.x] * matL[(threadIdx.x * blockDim.x)+i]);
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
		vecB[idx] = vecB[idx] - matL[(idx*numRows+i)- 1]*vecX[i - 1];

	if (idx == i)
	{
		vecX[i] = vecB[i] / matL[i*numRows+ i];
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
__global__ void gpu_square_solve_kernel_simple(int* matL, int* vecX, int* vecB, int numRows, int i)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int col_index = i * 2;
	if (col_index >= numRows)
		return;

	int row_index = idx;

	if (row_index < (i + 1) * 2 || row_index >= numRows)
		return;

	int value = matL[(row_index*numRows)+col_index]*vecB[col_index] + matL[(row_index*numRows)+col_index + 1]*vecB[col_index + 1];
	vecB[row_index] = vecB[row_index] - value;

}



void gpu_simple_solver(int* matL, int* vecX, int* vecB, int numRows)
{
	const unsigned int numThreadsPerBlock = N;
	//const unsigned int numBlocks = (numRows - 1) / numThreadsPerBlock + 1;
	const unsigned int numBlocks = 1;
	// Loop Below Executes 8 Times or for each Row of matL
	// for (int i = 0; i < numRows; i++)
		gpu_simple_solver_Anjum <<<numBlocks, numThreadsPerBlock >>>(matL, vecX, vecB, numRows);
}

void gpu_complex_solver(int* matL, int* vecX, int* vecB, int numRows)
{
	dim3 dimGrid((numRows / 2 - 1) / 32 + 1, (numRows / 2 - 1) / 32 + 1, 1);
	dim3 dimBlock(32, 32, 1);

	//gpu_square_update_kernel << <dimGrid, dimBlock >> >(matL, vecX, vecB, numRows);

	const unsigned int numThreadsPerBlock = N;
	//const unsigned int numBlocks = (numRows - 1) / numThreadsPerBlock + 1;
	const unsigned int numBlocks =  1;
	// Executed 4 Times Only should 8 times for each Row
	for (int i = 0; i < (numRows / 2); i++) 
	{
		gpu_square_solve_kernel_simple << <numBlocks, numThreadsPerBlock >> >(matL, vecX, vecB, numRows, i);
	}

	//copy B to X for the verification code in main.cu
	cudaMemcpy(vecX, vecB, numRows * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
}

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
	cuda_ret = cudaMemcpy(vecB_d,vecB_h, N*sizeof(int), cudaMemcpyHostToDevice);
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
	getchar();
    
}

int main()
{



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

	onDevice(matL_h, vecB_h, vecX_actual);



}
