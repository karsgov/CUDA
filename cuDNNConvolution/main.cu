#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <cudnn.h>

// Macro for checking status of cudnn objects
#define checkCUDNN(expression) {                               \
    cudnnStatus_t status = (expression);                       \
    if (status != CUDNN_STATUS_SUCCESS) {                      \
        std::cerr << "Error on line " << __LINE__ << ": "      \
                  << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                               \
    }                                                          \
}                                                              \

cv::Mat loadImage(const std::string imagePath) {
    cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

void saveImage(const std::string fname, float* buffer, int height, int width) {
    cv::Mat outputImage(height, width, CV_32FC3, buffer);
    // Clip negative values to zero
    cv::threshold(outputImage, 
                  outputImage,
                  /*threshold=*/0,
                  /*maxval=*/0,
                  cv::THRESH_TOZERO);
    cv::normalize(outputImage, outputImage, 0.0, 255.0, cv::NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8UC3);
    cv::imwrite(fname, outputImage);
    std::cout << "Saved image to " << fname << std::endl;
}
    
int main(int argc, char const *argv[]) {

    // Parse args
    if ( argc < 2) {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "\t" << argv[0] << " <image-path> [gpu=0]" << std::endl;
        return -1;
    }

    int gpuId = (argc > 2) ? std::atoi(argv[2]) : 0;

    // Load Image
    cv::Mat image = loadImage(argv[1]);

    // Create CUDNN context
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    
    // Describe input tensor
    cudnnTensorDescriptor_t inputDescriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputDescriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/3,
                                          /*image_height=*/image.rows,
                                          /*image_width=*/image.cols));

    // Describe kernel tensor
    cudnnFilterDescriptor_t kernelDescriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernelDescriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernelDescriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/3,
                                          /*in_channels=*/3,
                                          /*kernel_height=*/3,
                                          /*kernel_width=*/3));

    // Describe convolution kernel
    cudnnConvolutionDescriptor_t convolutionDescriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolutionDescriptor,
                                               /*pad_height=*/1,
                                               /*pad_width=*/1,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION));
                                               ///*computeType=*/CUDNN_DATA_FLOAT));

    // Get output image dimensions
    int batchSize{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolutionDescriptor,
                                                     inputDescriptor,
                                                     kernelDescriptor,
                                                     &batchSize,
                                                     &channels,
                                                     &height,
                                                     &width));

    // Describe output tensor
    cudnnTensorDescriptor_t outputDescriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&outputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDescriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/3,
                                          /*image_height=*/image.rows,
                                          /*image_width=*/image.cols));
    
    // Describe convolution operation
    cudnnConvolutionFwdAlgo_t convolutionAlgorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   inputDescriptor,
                                                   kernelDescriptor,
                                                   convolutionDescriptor,
                                                   outputDescriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   /*memoryLimitInBytes=*/0,
                                                   &convolutionAlgorithm));

    // Get workspace size
    size_t workspaceBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       inputDescriptor,
                                                       kernelDescriptor,
                                                       convolutionDescriptor,
                                                       outputDescriptor,
                                                       convolutionAlgorithm,
                                                       &workspaceBytes));

    std::cout << "Input image : " <<  image.channels() << " x " << image.rows 
              << " x " << image.cols << std::endl;
    std::cout << "Output image : " << channels << " x " << height << " x " 
              << width << std::endl;
    std::cout << "Workspace Size : " << (workspaceBytes/1048576.0) << " MB" << std::endl;


    // Allocate memory
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspaceBytes);

    int imageBytes = batchSize * channels * height * width * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc(&d_input, imageBytes);
    cudaMemcpy(d_input, image.ptr<float>(0), imageBytes, cudaMemcpyHostToDevice);

    float* d_output{nullptr};
    cudaMalloc(&d_output, imageBytes);
    cudaMemset(d_output, 0, imageBytes);

    // Define 2d kernel 
    const float kernelTemplate[3][3] = {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1}  
    };
    
    // Assign same kernel to differnt channels
    float h_kernel[3][3][3][3];

    for(int n=0; n < 3; ++n) {
        for(int c=0; c < 3; ++c) {
            for(int h=0; h < 3; ++h) {
                for(int w=0; w < 3; ++w) {
                    h_kernel[n][c][h][w] = kernelTemplate[h][w];
                }
            }
        }
    }

    float *d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);


    // Finally run convolution operation
    const float alpha=1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       inputDescriptor,
                                       d_input,
                                       kernelDescriptor,
                                       d_kernel,
                                       convolutionDescriptor,
                                       convolutionAlgorithm,
                                       d_workspace,
                                       workspaceBytes,
                                       &beta,
                                       outputDescriptor,
                                       d_output));

    // Copy output data from GPU to CPU memory
    float *h_output = new float[imageBytes];
    cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);

    // Save image
    saveImage("cuddn_out.png", h_output, height, width);

    // Free memory
    delete [] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    // Destroy desciptors
    cudnnDestroyTensorDescriptor(inputDescriptor);
    cudnnDestroyTensorDescriptor(outputDescriptor);
    cudnnDestroyFilterDescriptor(kernelDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);

    // Destroy cudnn context
    cudnnDestroy(cudnn);

    return 0;
}
