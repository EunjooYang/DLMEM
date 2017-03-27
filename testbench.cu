/*
   * Writer : Eunjoo Yang
   *
   * Testing the effect of memory layout in GPU
   * cuDNN vs cuda-convnet
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cudnn.h>

#define NCHW 0
#define CHWN 1
#define TRUE 1
#define FALSE 0
#define MODE NCHW 
#define DEBUG TRUE
#define CHECK(name,func) printf("\t%s: %s\n",name,cudnnGetErrorString(func))


int main(){

    
// IF CUDNN MODE
#if MODE==NCHW

    // Test Parameter
    cudnnHandle_t cudnnHandler;
    cudnnTensorDescriptor_t tensorDesc;
    double *d_ts, *h_ts;
    int input_n = 10;
    int input_c = 3;
    int input_h = 28;
    int input_w = 28;

    // Initialize host and device memory
    h_ts = (doube *)malloc(n*c*h*w*sizeof(double));
    cudaMalloc((void **)&d_ts, n*c*h*w*sizeof(double));

#if DEBUG==TRUE

    // initialize the cuDNN library and creates a handle to an opaque structure
    // holding the cuDNN library context
    // cuDNN library context is tied to the current CUDA device
    CHECK("cudnnCreate",cudnnCreate(&cudnnHandler));
    // Initialize Generic Tensor Descriptor first
    CHECK("CreateTensorDescriptor", cudnnCreateTensorDescriptor(&tensorDesc));
    // Create TensorDescriptor for 4D Descriptor
    CHECK("TensorDescriptor",cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, input_n, input_c,input_h, input_w));


    // free host memory
    free(h_ts);
    // free device memory
    cudaFree(d_ts);
    // Destroy Tensor Descriptor
    CHECK("Destroy TensorDescriptor",cudnnDestroyTensorDescriptor(tensorDesc));
    // Destroy cudnnHandler
    CHECK("cudnnDestroy",cudnnDestroy(cudnnHandler));
#else


#endif
#endif
}
