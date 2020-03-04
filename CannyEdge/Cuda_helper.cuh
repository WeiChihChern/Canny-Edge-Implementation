#pragma once
#include <cuda_runtime.h>


// Marco for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}


// Init API
void cuda_init()
{
	cudaFree(0);
}

#endif