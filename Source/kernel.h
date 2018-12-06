

#include <cuda_runtime.h>
#define N 8
//numRows = numCols since L is square
//               y,x
#define L_Matrix(row,col) matL[((row)*numRows + (col))]

void gpu_simple_solver(int* matL, int* vecX, int* vecB, int numRows);
void gpu_complex_solver(int* matL, int* vecX, int* vecB, int numRows);

/*
gpu_simple_solver(matL_d, vecX_d, vecB_d, N);
cudaDeviceSynchronize();


gpu_complex_solver(matL_d, vecX_d,  vecB_d, N);
*/