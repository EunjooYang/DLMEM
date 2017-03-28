/*
   * Writer : Eunjoo Yang
   *
   * Testing the effect of memory layout in GPU
   * cuDNN vs cuda-convnet
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cudnn.h>  // cudnn
#include <curand.h> // curand

#define NCHW 0
#define CHWN 1
#define TRUE 1
#define FALSE 0
#define MODE NCHW 
#define DEBUG TRUE
#define CHECK(name,func) printf("\t%s: %s\n",name,cudnnGetErrorString(func))
#define outputDim(inputDim,pad,filterDim,convolutionStride) 1+(inputDim + 2*pad - filterDim)/convolutionStride
#define FOWARDALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM


int main(){

    
// IF CUDNN MODE
#if MODE==NCHW


    curandGenerator_t gen;  //curand generator
    cudnnHandle_t cudnnHandler;
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    float *d_input, *h_input;
    float *d_output, *h_output;
    float *d_w; 

    int input_n = 10;
    int input_c = 3;
    int input_h = 28;
    int input_w = 28;
    int filter_cout = 10;
    int filter_cin = input_c;
    int filter_height = 3;
    int filter_width = 3;
    int conv_pad[2] = {0, 0};
    int conv_stride[2] = {1,1};
    int out_w = outputDim(input_w,conv_pad[0],filter_width,conv_stride[0]);
    int out_h = outputDim(input_h,conv_pad[1],filter_height,conv_stride[1]);
    size_t wsSize;
    unsigned int *d_workspace;
    float alpha = 1;
    float beta = 0;


    // Allocate host memory
    h_input = (float*)malloc(input_n*input_c*input_h*input_w*sizeof(float));
    h_output = (float*)malloc(out_w*out_h*filter_cout*input_n*sizeof(float));

    // Allocate device memory
    cudaMalloc((void**) &d_input, input_n*input_c*input_h*input_w*sizeof(float));
    cudaMalloc((void**) &d_output, out_w*out_h*filter_cout*input_n*sizeof(float));
    cudaMalloc((void**) &d_w, filter_cout*filter_height*filter_width*sizeof(float));


// DEBUG MODE : PRINT STATE MESSAGE
#if DEBUG==TRUE

    // Initialize with Random Value in device memory
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
    curandGenerate(gen,(unsigned int*)d_input,input_n*input_c*input_h*input_w);
    curandGenerate(gen,(unsigned int*)d_w,filter_cout*filter_height*filter_width);

    // initialize the cuDNN library and creates a handle to an opaque structure
    // holding the cuDNN library context
    // cuDNN library context is tied to the current CUDA device
    CHECK("cudnnCreate",cudnnCreate(&cudnnHandler));
    // Initialize Generic Tensor Descriptor first
    CHECK("CreateTensorDescriptor", cudnnCreateTensorDescriptor(&inputDesc));
    // Create TensorDescriptor for 4D Descriptor
    CHECK("TensorDescriptor",cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c,input_h, input_w));
    // Initialize Generic Tensor Descriptor first
    CHECK("CreateTensorDescriptor", cudnnCreateTensorDescriptor(&outputDesc));
    // Create TensorDescriptor for 4D Descriptor
    CHECK("TensorDescriptor",cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, filter_cout,out_h, out_w));
    // Create Filter Descriptor
    CHECK("CreateFilterDescriptor", cudnnCreateFilterDescriptor(&filterDesc));
    // SetFilter4dDescriptor
    CHECK("SetFilterDescriptor4D",cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_cout, filter_cin, filter_height, filter_width));
    // Create Convolution Descriptor
    CHECK("CreateconvolutionDescriptor",cudnnCreateConvolutionDescriptor(&convDesc));
    // Set Convolution 2d Descriptor
    CHECK("SetConvolutionDescriptor2D",cudnnSetConvolution2dDescriptor(convDesc,conv_pad[0],conv_pad[1],conv_stride[0],conv_stride[1],1,1,CUDNN_CONVOLUTION));

    // Get Convolution Forward Workspace Size
    CHECK("Get Workspace Size",cudnnGetConvolutionForwardWorkspaceSize(cudnnHandler,inputDesc,filterDesc,convDesc,outputDesc,FOWARDALGO,&wsSize));
    // Generate Workspace Size
    cudaMalloc((void**) &d_workspace, wsSize);
    // Convolution Forward
    CHECK("cudnnConvolutionForward",cudnnConvolutionForward(cudnnHandler,&alpha,inputDesc,d_input,filterDesc,d_w,convDesc,FOWARDALGO,d_workspace,wsSize,&beta,outputDesc,d_output));
    cudaDeviceSynchronize();
    // free host memory
    free(h_input);
    free(h_output);
    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_w);
    cudaFree(d_workspace);

    // Destroy Filter Descriptor
    CHECK("DestroyFilterDescriptor", cudnnDestroyFilterDescriptor(filterDesc));
    // Destroy Tensor Descriptor
    CHECK("Destroy TensorDescriptor",cudnnDestroyTensorDescriptor(inputDesc));
    // Destroy cudnnHandler
    CHECK("cudnnDestroy",cudnnDestroy(cudnnHandler));
    // Destroy Curand Generator
    curandDestroyGenerator(gen);
#else
#endif
#endif
}
