#pragma once


#include "Edge_Cuda.cuh"
#include "Cuda_helper.cuh"
#include <math.h>





// cudaReadModeNormalizedFloat = read elements in normalized float
// cudaReadModeElementType     = Read texture as specified element type
texture <float, cudaTextureType2D, cudaReadModeElementType>     ker_h_tex;
texture <float, cudaTextureType2D, cudaReadModeElementType>     ker_v_tex;
texture <uchar, cudaTextureType2D, cudaReadModeNormalizedFloat>		canny_tex;

cudaArray *d_canny_src = 0;   // input data, will be binded to texture
cudaArray *d_ker_h = 0; // input kernel, will be binded to texture
cudaArray *d_ker_v = 0; // input kernel, will be binded to texture

float sobel_horizontal[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
float sobel_vertical[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

float *gpu_dst_v, *gpu_dst_h;
uchar *non_max;







__global__
void conv2_h(
	float *dst, 
	int rows, 
	int cols,
	int ker_rows, 
	int ker_cols)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < cols && y < rows)
	{
		float sum = 0.0f;

		#pragma unroll
		for (int i = -ker_cols / 2; i < ker_cols / 2 + 1; i++)
			for (int j = -ker_rows / 2; j < ker_rows / 2 + 1; j++)
				sum += tex2D(canny_tex, x + i, y + j) * tex2D(ker_h_tex, i + 1, j + 1);
			
		dst[y*cols + x] = sum * 255; // Scale back to uchar's range
	}
};


__global__
void conv2_v(
	float *dst,
	int rows,
	int cols,
	int ker_rows,
	int ker_cols)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < cols && y < rows)
	{
		float sum = 0.0f;

		#pragma unroll
		for (int i = -ker_cols / 2; i < ker_cols / 2 + 1; i++)
			for (int j = -ker_rows / 2; j < ker_rows / 2 + 1; j++)
				// tex value will be normalized to 0.0~1.0 in float
				sum += tex2D(canny_tex, x + i, y + j) * tex2D(ker_v_tex, i + 1, j + 1);

		dst[y*cols + x] = sum * 255; // Scale back 
	}
};



__device__ __forceinline__
float get_magnitude(float src1, float src2)
{
	return sqrt(src1 * src1 + src2 * src2);
};

__device__ __forceinline__
float get_edgeDrections(float gy, float gx)
{
	float w = abs(gy / (gx + 0.0001));

	if (w < 0.4)
		return 0.0f;
	else if (w > 2.3)
		return 90.0f;
	else
		return (gx*gy > 0) ? -45.0f : 45.0f; 
};

__global__
void get_info_from_edge(float* src1, float* src2, int rows, int cols)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	int idx = y*cols + x;
	if (x < cols && y < rows)
	{

		// Temporary store results, since second calculation needs both src1, src2 untouched
		float temp_mag = get_magnitude(src1[idx], src2[idx]);
		float temp_dir = get_edgeDrections(src1[idx], src2[idx]);

		// using src1 to store magnitude result & src2 to store direction result
		src1[idx] = temp_mag;
		src2[idx] = temp_dir;
	}
};


// Cuda store 2D data in column major order
// Opencv does the otherwise
__global__
void nonMax(
	float* mag, float* gra, uchar* dst,
	int rows, int cols,
	float h_thres, float l_thres)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;


	int idx = y*cols + x;
	if (x > 1 && x < cols && y < rows && y > 1)
	{
		float cur_mag_val = mag[idx];
		float direction = gra[idx];
		if (cur_mag_val > l_thres && cur_mag_val != 0)
		{
			if (direction == 90)
			{
				if (cur_mag_val > mag[idx - 1] && cur_mag_val >= mag[idx + 1])
					dst[idx] = (cur_mag_val >= h_thres) ? 255 : 125;
				
			}
			else if (direction == 0)
			{
				if (cur_mag_val > mag[idx - cols] && cur_mag_val >= mag[idx + cols])
					dst[idx] = (cur_mag_val >= h_thres) ? 255 : 125;
			}
			else
			{
				int d = (direction == 45) ? 1 : -1;
				if (cur_mag_val >= mag[idx + cols - d] && cur_mag_val > mag[idx - cols + d])
					dst[idx] = (cur_mag_val >= h_thres) ? 255 : 125;
			}
		}
		else
			dst[idx] = 0;
	}
}








__host__ __forceinline__
void src_kernel_init(uchar *src, int rows, int cols)
{
	uint     size = cols * rows * sizeof(uchar);
	uint ker_size = 3 * 3 * sizeof(float);

	// allocate array and copy image data
	cudaChannelFormatDesc channelDesc     = cudaCreateChannelDesc(8,  0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc channelDesc_ker1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc_ker2 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	gpuErrchk(cudaMallocArray(&d_canny_src, &channelDesc, cols, rows)); // src image
	gpuErrchk(cudaMallocArray(&d_ker_h, &channelDesc_ker1, 3, 3)); // kernel horizontal
	gpuErrchk(cudaMallocArray(&d_ker_v, &channelDesc_ker2, 3, 3)); // kernel horizontal

	gpuErrchk(cudaMemcpyToArray(d_canny_src, 0, 0, src, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToArray(d_ker_h, 0, 0, sobel_horizontal, ker_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToArray(d_ker_v, 0, 0, sobel_vertical, ker_size, cudaMemcpyHostToDevice));

}



__host__ __forceinline__ 
void textureSetUp()
{

	// set texture parameters
	canny_tex.addressMode[0] = cudaAddressModeMirror;
	canny_tex.addressMode[1] = cudaAddressModeMirror;
	canny_tex.filterMode = cudaFilterModePoint;      // no interpolation between pixels like pixel at (1.5, 1.5)
	canny_tex.normalized = false;                    // access texture with not-normalized coordinates
	gpuErrchk(cudaBindTextureToArray(canny_tex, d_canny_src));

	ker_h_tex.addressMode[0] = cudaAddressModeMirror;
	ker_h_tex.addressMode[1] = cudaAddressModeMirror;
	ker_h_tex.filterMode = cudaFilterModePoint;      // no interpolation between pixels like pixel at (1.5, 1.5)
	ker_h_tex.normalized = false;                    // access texture with not-normalized coordinates
	gpuErrchk(cudaBindTextureToArray(ker_h_tex, d_ker_h));

	ker_v_tex.addressMode[0] = cudaAddressModeMirror;
	ker_v_tex.addressMode[1] = cudaAddressModeMirror;
	ker_v_tex.filterMode = cudaFilterModePoint;      // no interpolation between pixels like pixel at (1.5, 1.5)
	ker_v_tex.normalized = false;                    // access texture with not-normalized coordinates
	gpuErrchk(cudaBindTextureToArray(ker_v_tex, d_ker_v));

}


__host__ __forceinline__
void memClean()
{
	cudaUnbindTexture(canny_tex);
	cudaUnbindTexture(ker_h_tex);
	cudaUnbindTexture(ker_v_tex);
	cudaFreeArray(d_canny_src);
	cudaFreeArray(d_ker_h);
	cudaFreeArray(d_ker_v);
	cudaFree(gpu_dst_v);
	cudaFree(gpu_dst_h);
	cudaFree(non_max);
}




extern "C"
void canny_cuda_impl(
	uchar*src, uchar *dst,
	int rows, int cols,
	float high_thres, float low_thres)
{
	// cudaMallocArray & cudaMemcpyToArray for input src & sobel kernels
	src_kernel_init(src, rows, cols); 
	// bind src & kernels to textures
	textureSetUp();

	// Create gpu memory to store some result
	int s_float = rows*cols * sizeof(float);
	int s_uchar = rows*cols * sizeof(uchar);
	gpuErrchk(cudaMalloc((void **)&gpu_dst_v, s_float)); // store sobel vertical convolution result
	gpuErrchk(cudaMalloc((void **)&gpu_dst_h, s_float)); // store sobel horizontal convlution result
	gpuErrchk(cudaMalloc((void **)&non_max, s_uchar));   // store non max suppression result in uchar 


	dim3 threadPerBlock(8, 8);
	dim3 blocksPerGrid((cols / threadPerBlock.x) + 1, (rows / threadPerBlock.y) + 1);

	// Convolution with 2 sobel kernels
	conv2_h <<<blocksPerGrid, threadPerBlock>>> (gpu_dst_h, rows, cols, 3, 3);
	conv2_v <<<blocksPerGrid, threadPerBlock>>> (gpu_dst_v, rows, cols, 3, 3);



	// Bind gpu_dst_h & _v to texture




	// Get magnitude and direction result from 2 edge maps
	//	Then reuse the variable:
	//		gpu_dst_h will be replaced with magnitdue data
	//		gpu_dst_v will be replaced with direction data
	get_info_from_edge <<<blocksPerGrid, threadPerBlock>>>(gpu_dst_h, gpu_dst_v, rows, cols);

	// Performance non maximum suppression & leave the hysteresis thresholding to cpu for now
	nonMax<<<blocksPerGrid, threadPerBlock >> >(gpu_dst_h, gpu_dst_v, non_max, rows, cols, high_thres, low_thres);


	gpuErrchk(cudaMemcpy(dst, non_max, s_uchar, cudaMemcpyDeviceToHost));


	memClean();

	return;
}




#endif // !_EDGE_CUDA_CU_
