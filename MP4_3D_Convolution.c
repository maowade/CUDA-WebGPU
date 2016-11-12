#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }               \
  } while (0)

//@@ Define any useful program-wide constants here

//@@ Define constant memory for device kernel here
#define Mask_Width 3
#define O_TILE_WIDTH 8
#define Mask_Radius 1
#define TILE_WIDTH 10
__constant__ float M[Mask_Width][Mask_Width][Mask_Width];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  int x_o = blockIdx.x * O_TILE_WIDTH + tx;
  int y_o = blockIdx.y * O_TILE_WIDTH + ty;
  int z_o = blockIdx.z * O_TILE_WIDTH + tz;
  
  int x_i = x_o-(Mask_Radius);
  int y_i = y_o-(Mask_Radius);
  int z_i = z_o-(Mask_Radius);
  
  __shared__ float Ns[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  if((x_i >= 0) && (x_i < x_size) && (y_i >= 0) && (y_i < y_size) && (z_i >= 0) && (z_i < z_size)){
    
    Ns[tz][ty][tx] = input[x_size * y_size * z_i + x_size * y_i + x_i];
    
  }
  else{
    
    Ns[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  
  float Val = 0.0f;
  if(tx < O_TILE_WIDTH && ty < O_TILE_WIDTH && tz < O_TILE_WIDTH){
    
    for(int i = 0; i < Mask_Width; i++){
      for(int j = 0; j < Mask_Width; j++){
        for(int k = 0; k < Mask_Width; k++){
          
          Val += M[i][j][k] * Ns[i + tz][j + ty][k + tx];
        }
      }
    }
    
    
    if(x_o < x_size && y_o < y_size && z_o < z_size){
      output[x_size * y_size * z_o + x_size * y_o + x_o] = Val;
    }
  }
  
  __syncthreads();
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  
      wbCheck(cudaMalloc((void **) &deviceInput, x_size * y_size * z_size * sizeof(float)));
      wbCheck(cudaMalloc((void **) &deviceOutput, x_size * y_size * z_size * sizeof(float)));
      
      //cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, &hostInput[3],(inputLength - 3) * sizeof(float),cudaMemcpyHostToDevice ));//**
  wbCheck(cudaMemcpyToSymbol(M, hostKernel, (Mask_Width * Mask_Width * Mask_Width) * sizeof(float)));
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size / 8.0), ceil(y_size / 8.0),ceil(z_size / 8.0));//**
  dim3 DimBlock(10, 10, 10);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  //wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput,(x_size * y_size * z_size)*sizeof(float),cudaMemcpyDeviceToHost));
          cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

