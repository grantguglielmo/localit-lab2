#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < N; index += blockDim.x*gridDim.x){
       result[index] = alpha * x[index] + y[index];
    }
}

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary

    float *device_x;
    float *device_y;
    float *device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    cudaMalloc(&device_x, total_elems*sizeof(float));
    cudaMalloc(&device_y, total_elems*sizeof(float));
    cudaMalloc(&device_result, total_elems*sizeof(float));
    long size = total_elems/partitions;
    cudaStream_t streams[partitions];

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    for (int i=0; i<partitions; i++) {
  
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //
        cudaStreamCreate(&streams[i]);
        long index = size*i;
        cudaMemcpy(device_x + index, xarray + index, size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_y + index, yarray + index, size*sizeof(float), cudaMemcpyHostToDevice);
         
        //
        // TODO: insert time here to begin timing only the kernel
        //
        double startGPUTime = CycleTimer::currentSeconds();
    
        // compute number of blocks and threads per block
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        // run saxpy_kernel on the GPU
        saxpy_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(size, alpha, device_x + index, device_y + index, device_result + index);
    
        //
        // TODO: insert timer here to time only the kernel.  Since the
        // kernel will run asynchronously with the calling CPU thread, you
        // need to call cudaDeviceSynchronize() before your timer to
        // ensure the kernel running on the GPU has completed.  (Otherwise
        // you will incorrectly observe that almost no time elapses!)
        //
        cudaStreamSynchronize(streams[i]);
        double endGPUTime = CycleTimer::currentSeconds();
        double timeKernel = endGPUTime - startGPUTime;
    
        cudaError_t errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }
    
        //
        // TODO: copy result from GPU using cudaMemcpy
        //
        cudaMemcpy(resultarray + index, device_result + index, size*sizeof(float), cudaMemcpyDeviceToHost);
    }

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;

    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
