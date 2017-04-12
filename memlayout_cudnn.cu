/*
   * memlayout_cudnn.cu
   * 
   * Writer : Eunjoo Yang
   *
   * Testing the effect of memory layout in GPU
   * This code only test the cudnn (NCHW)
   * 
   * How to execute ./memlayout_cudnn minibatchsize input_c input_size output_c filter_size
   *
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cudnn.h>  // cudnn
#include <curand.h> // curand
#include <sys/time.h> // get time of day

#define NCHW 0
#define CHWN 1
#define TRUE 1
#define FALSE 0
#define MODE NCHW 
#define DEBUG FALSE
// MACRO for debugging
#define CHECK(name,func) printf("\t%s: %s\n",name,cudnnGetErrorString(func))    
// MACRO to calculate output dimension of filter
#define outputDim(inputDim,pad,filterDim,convolutionStride) 1+(inputDim + 2*pad - filterDim)/convolutionStride
// convolution method
#define FOWARDALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM


int main(int argc, char *argv[]){

    
    curandGenerator_t gen;  //curand generator
    cudnnHandle_t cudnnHandler;
    cudnnTensorDescriptor_t inputDesc;
    cudnnTensorDescriptor_t outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    float *d_input, *h_input;
    float *d_output, *h_output;
    float *d_w; 

    int input_n = 256;
    int input_c = 3;
    int input_h = 24;
    int input_w = 24;
    int filter_cout = 64;
    int filter_cin = input_c;
    int filter_height = 5;
    int filter_width = 5;
    int conv_pad[2] = {0, 0};
    int conv_stride[2] = {1,1};
    int out_w = outputDim(input_w,conv_pad[0],filter_width,conv_stride[0]);
    int out_h = outputDim(input_h,conv_pad[1],filter_height,conv_stride[1]);
    size_t wsSize;
    unsigned int *d_workspace;
    float alpha = 1;
    float beta = 0;
    struct timeval start_point, end_point;
    double elapsed_time;

    if (argc < 6){
        printf("Execution Error\n");
        printf("Please Enter the Execution Options \n");
        printf("./memlayout_cudnn 'minibatchsize' 'input_c' 'input_size' 'output_c' 'filter_size'\n");
        printf("Follow Default Setting\n");
        printf("======================================\n");
        printf("Start to cudnn memory layout test\n");
        printf("\t mini batch size : %d\n",input_n);
        printf("\t input channel : %d\n", input_c);
        printf("\t input image size: %d\n", input_w);
        printf("\t output channel : %d\n", filter_cout);
        printf("\t filter size : %d\n", filter_height);
        printf("======================================\n");
    }else{

        input_n = atoi(argv[1]);  
        input_c = atoi(argv[2]);
        input_w = input_h =  atoi(argv[3]);
        filter_cout = atoi(argv[4]);
        filter_height = filter_width = atoi(argv[5]);
        printf("======================================\n");
        printf("Start to cudnn memory layout test\n");
        printf("\t mini batch size : %d\n",input_n);
        printf("\t input channel : %d\n", input_c);
        printf("\t input image size: %d\n", input_w);
        printf("\t output channel : %d\n", filter_cout);
        printf("\t filter size : %d\n", filter_height);
        printf("======================================\n");
    }

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

    // measure the time
    gettimeofday(&start_point,NULL);
    // Convolution Forward
    CHECK("cudnnConvolutionForward",cudnnConvolutionForward(cudnnHandler,&alpha,inputDesc,d_input,filterDesc,d_w,convDesc,FOWARDALGO,d_workspace,wsSize,&beta,outputDesc,d_output));
    // wait until done
    cudaDeviceSynchronize(); 
    // measure the finish time
    gettimeofday(&end_point,NULL);
    elapsed_time = (double)(end_point.tv_sec)*1000+(double)(end_point.tv_usec)/1000-(double)(start_point.tv_sec)*1000-(double)(start_point.tv_usec)/1000;
    printf("Elapsed Time : %f ms\n",elapsed_time);


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

    // Initialize with Random Value in device memory
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
    curandGenerate(gen,(unsigned int*)d_input,input_n*input_c*input_h*input_w);
    curandGenerate(gen,(unsigned int*)d_w,filter_cout*filter_height*filter_width);

    // initialize the cuDNN library and creates a handle to an opaque structure
    // holding the cuDNN library context
    // cuDNN library context is tied to the current CUDA device
    cudnnCreate(&cudnnHandler);
    // Initialize Generic Tensor Descriptor first
    cudnnCreateTensorDescriptor(&inputDesc);
    // Create TensorDescriptor for 4D Descriptor
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c,input_h, input_w);
    // Initialize Generic Tensor Descriptor first
    cudnnCreateTensorDescriptor(&outputDesc);
    // Create TensorDescriptor for 4D Descriptor
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, filter_cout,out_h, out_w);
    // Create Filter Descriptor
    cudnnCreateFilterDescriptor(&filterDesc);
    // SetFilter4dDescriptor
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_cout, filter_cin, filter_height, filter_width);
    // Create Convolution Descriptor
    cudnnCreateConvolutionDescriptor(&convDesc);
    // Set Convolution 2d Descriptor
    cudnnSetConvolution2dDescriptor(convDesc,conv_pad[0],conv_pad[1],conv_stride[0],conv_stride[1],1,1,CUDNN_CONVOLUTION);

    // Get Convolution Forward Workspace Size
    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandler,inputDesc,filterDesc,convDesc,outputDesc,FOWARDALGO,&wsSize);
    // Generate Workspace Size
    cudaMalloc((void**) &d_workspace, wsSize);

    // measure the time
    gettimeofday(&start_point,NULL);
    // Convolution Forward
    cudnnConvolutionForward(cudnnHandler,&alpha,inputDesc,d_input,filterDesc,d_w,convDesc,FOWARDALGO,d_workspace,wsSize,&beta,outputDesc,d_output);
    // wait until done
    cudaDeviceSynchronize(); 
    // measure the finish time
    gettimeofday(&end_point,NULL);
    elapsed_time = (double)(end_point.tv_sec)*1000+(double)(end_point.tv_usec)/1000-(double)(start_point.tv_sec)*1000-(double)(start_point.tv_usec)/1000;
    printf(" Elapsed Time : %f ms\n",elapsed_time);


    // free host memory
    free(h_input);
    free(h_output);
    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_w);
    cudaFree(d_workspace);

    // Destroy Filter Descriptor
    cudnnDestroyFilterDescriptor(filterDesc);
    // Destroy Tensor Descriptor
    cudnnDestroyTensorDescriptor(inputDesc);
    // Destroy cudnnHandler
    cudnnDestroy(cudnnHandler);
    // Destroy Curand Generator
    curandDestroyGenerator(gen);
#endif
}
