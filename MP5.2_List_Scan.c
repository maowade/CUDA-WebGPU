#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {													\
	cudaError_t err = stmt;											   \
	if (err != cudaSuccess) {											 \
		wbLog(ERROR, "Failed to run stmt ", #stmt);					   \
		wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
		return -1;														\
	}																	 \
} while(0)

__global__ void update(const float *endpoints, float *output, int len) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ((blockIdx.x > 0) && (i < len)) output[i] += endpoints[blockIdx.x-1];
}
	
__global__ void scan(const float * input, float * output, float *endpoints, int len) {
	//@@ Modify the body of this function to complete the functionality of
	//@@ the scan on the device
	//@@ You may need multiple kernel calls; write your kernels before this
	//@@ function and call them from here
	__shared__ float T[BLOCK_SIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) T[threadIdx.x] = input[i];
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		// We start at 2 * stride - 1 and for each threadIdx.x we add a 2*stride
		int index = (2 * stride - 1) + (threadIdx.x * stride * 2);
		if (index < BLOCK_SIZE) T[index] += T[index-stride];
		__syncthreads();
	}

	for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
		int index = (3 * stride - 1) + (threadIdx.x * 2 * stride);
		if (index < BLOCK_SIZE) T[index] += T[index-stride];
		__syncthreads();
	}
	if (i < len) output[i] = T[threadIdx.x];
	__syncthreads();
	if ((endpoints != NULL) && (threadIdx.x == 0)) endpoints[blockIdx.x] = T[blockDim.x-1];
}

int main(int argc, char ** argv) {
	wbArg_t args;
	float * hostInput; // The input 1D list
	float * hostOutput; // The output list
	float * deviceInput;
	float * deviceOutput;
	float * deviceEndpoints;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
	hostOutput = (float*) malloc(numElements * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((((numElements-1) / BLOCK_SIZE)+1), 1, 1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Modify this to complete the functionality of the scan
	wbCheck(cudaMalloc((void **)&deviceEndpoints, sizeof(float) * dimGrid.x));
	//@@ on the deivce
	scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceEndpoints, numElements);
	scan<<<dimGrid, dimBlock>>>(deviceEndpoints, deviceEndpoints, NULL, dimGrid.x-1);
	update<<<dimGrid, dimBlock>>>(deviceEndpoints, deviceOutput, numElements);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceEndpoints);
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	free(hostOutput);

	return 0;
}

