#define VKFFT_BACKEND 1

#include <iostream>
#include <cufftw.h>
#include <chrono>
#include <vkFFT.h>

struct interfaceFFTPlan {
    VkFFTConfiguration *config;
    VkFFTApplication *app;
    VkFFTLaunchParams *lParams;
    bool isBaked;
    bool notInit;
    CUdevice device;
    CUcontext context;
    int dataType;
    int device_id;
    uint64_t inputBufferSize;
    uint64_t outputBufferSize;
};

typedef enum vkfft_transform_dir {
    VKFFT_FORWARD_TRANSFORM = -1,
    VKFFT_BACKWARD_TRANSFORM = 1
} vkfft_transform_dir;

typedef struct interfaceFFTPlan interfaceFFTPlan;


VkFFTResult
performVulkanFFT(interfaceFFTPlan *plan, vkfft_transform_dir inverse, uint64_t num_iter) {
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cudaError_t res = cudaSuccess;
    std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_iter; i++) {
        resFFT = VkFFTAppend(plan->app, inverse, plan->lParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
    }
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
    std::cout << totTime << std::endl;
    return resFFT;
}

VkFFTResult
performVulkanFFTiFFT(interfaceFFTPlan *plan, uint64_t num_iter) {
    VkFFTResult resFFT = VKFFT_SUCCESS;

    cudaError_t res = cudaSuccess;
    std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_iter; i++) {
        resFFT = VkFFTAppend(plan->app, vkfft_transform_dir::VKFFT_FORWARD_TRANSFORM, plan->lParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
        resFFT = VkFFTAppend(plan->app, vkfft_transform_dir::VKFFT_BACKWARD_TRANSFORM, plan->lParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
    }
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
    std::cout << totTime << std::endl;

    return resFFT;
}


//vkfftPlanMany
VkFFTResult
vkfftPlanMany(interfaceFFTPlan *interface, int rank, int *doubleComplex,
              int *doubleComplexPadded, int istride,
              int doubleComplexPaddedTotal, int *complexGridSizePadded, int ostride,
              int complexGridSizePaddedTotal, cufftType type, int batch, cudaStream_t *stream) {
    cuDoubleComplex *buffer = 0;
    cuDoubleComplex *buffer1 = 0;

    const int ZZ = 3, XX = 1, YY = 2;
    interface->config->FFTdim = 3;
    interface->config->size[0] = doubleComplex[ZZ];
    interface->config->size[1] = doubleComplex[XX];
    interface->config->size[2] = doubleComplex[YY];
    interface->config->doublePrecision = true;
    //configuration.disableMergeSequencesR2C = 1;
    interface->config->device = (CUdevice *) malloc(sizeof(CUdevice));
    cudaError_t result = cudaGetDevice(interface->config->device);
    if (result != cudaSuccess) {
        printf("VKFFT_ERROR_FAILED_TO_GET_DEVICE error: %d\n", result);
        return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    }
    interface->config->num_streams = 1;
    interface->config->stream = stream;

    uint64_t bufferSize =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ] *
            sizeof(cufftDoubleComplex);
    std::cout << "bufferSize = " << bufferSize << std::endl;
    interface->config->bufferSize = &bufferSize;
    interface->config->bufferStride[0] = complexGridSizePadded[ZZ];
    interface->config->bufferStride[1] = complexGridSizePadded[ZZ] * complexGridSizePadded[YY];
    interface->config->bufferStride[2] =
            complexGridSizePadded[ZZ] * complexGridSizePadded[YY] * complexGridSizePadded[XX];
    result = cudaMalloc((void **) &buffer, bufferSize);
    if (result != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    interface->config->buffer = (void **) &buffer;


    interface->config->isInputFormatted = 1;
    interface->config->inverseReturnToInputBuffer = 1;
    uint64_t inputBufferSize =
            doubleComplexPadded[XX] * doubleComplexPadded[YY] * doubleComplexPadded[ZZ] * sizeof(cufftDoubleComplex);
    std::cout << "inputBufferSize = " << inputBufferSize << std::endl;
    interface->config->inputBufferSize = &inputBufferSize;
    interface->config->inputBufferStride[0] = doubleComplexPadded[ZZ];
    interface->config->inputBufferStride[1] = doubleComplexPadded[ZZ] * doubleComplexPadded[YY];
    interface->config->inputBufferStride[2] =
            doubleComplexPadded[ZZ] * doubleComplexPadded[YY] * doubleComplexPadded[XX];
    result = cudaMemcpy(buffer, buffer1, inputBufferSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
    interface->config->inputBuffer = (void **) &buffer1;
    VkFFTResult resFFT = initializeVkFFT(interface->app, *interface->config);
    if (resFFT != VKFFT_SUCCESS) printf("VkFFT error: %d\n", resFFT);


    std::cout << "vkFFT: complex dim = " << doubleComplexPadded[XX] << "x" << doubleComplexPadded[YY] << "x"
              << doubleComplexPadded[ZZ] << std::endl;
    return resFFT;
}

VkFFTResult vkfftExecZ2Z(interfaceFFTPlan *inerface,
                         cufftDoubleComplex *idata,
                         cufftDoubleComplex *odata,
                         int direction, cudaStream_t *stream) {
    uint64_t num_iter = (((uint64_t) 4096 * 1024.0 * 1024.0) / *(inerface->config->bufferSize) > 1000) ? 1000 :
                        (uint64_t) ((uint64_t) 4096 * 1024.0 * 1024.0) / *inerface->config->bufferSize;
    auto resFFT = performVulkanFFTiFFT(inerface, num_iter);
    return resFFT;
}

VkFFTResult vkfftBakeFFTPlan(interfaceFFTPlan *plan, cudaStream_t *stream) {
    VkFFTResult res;
#if(__DEBUG__ > 0)
    printf("Begin initialization...\n");
#endif
    // If the plan was baked previously, the previous plan needs to be deleted
    if ((plan->app != NULL) && (plan->isBaked)) {
        deleteVkFFT(plan->app);
        plan->app = (VkFFTApplication *) calloc(1, sizeof(VkFFTApplication));
    }
    VkFFTConfiguration tmpConfig = *plan->config;
    res = initializeVkFFT(plan->app, tmpConfig);
#if(__DEBUG__ > 0)
    printf("    Done with initialization...\n");
#endif

    CUresult cuda_res = CUDA_SUCCESS;
    cudaError_t cuda_res2 = cudaSuccess;
    std::cout << "First Test" << std::endl;

    cuda_res = cuInit(0);
    if (cuda_res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    cuda_res2 = cudaSetDevice((int) plan->device_id);
    if (cuda_res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
    cuda_res = cuDeviceGet(&plan->device, (int) plan->device_id);
    if (cuda_res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    cuda_res = cuCtxCreate(&plan->context, 0, (int) plan->device);
    if (cuda_res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;

    if (res == VKFFT_SUCCESS) {
        plan->isBaked = true;
    } else {
        plan->isBaked = false;
    }
    plan->notInit = true;
    return res;
}

int main() {
    interfaceFFTPlan *plan_interface = {};

    plan_interface = (interfaceFFTPlan *) malloc(sizeof(interfaceFFTPlan));
    plan_interface->config = (VkFFTConfiguration *) malloc(sizeof(VkFFTConfiguration));
    plan_interface->app = (VkFFTApplication *) malloc(sizeof(VkFFTApplication));
    plan_interface->lParams = (VkFFTLaunchParams *) malloc(sizeof(VkFFTLaunchParams));

    cudaStream_t *t;


    std::cout << "Second Test" << std::endl;
    printf("12 - VkFFT/FFTW C2C precision test in double precision\n");
    const int num_benchmark_samples = 1;
    const int num_runs = 1;

    uint64_t benchmark_dimensions[num_benchmark_samples][4] = {
            {(uint64_t) pow(2, 8), (uint64_t) pow(2, 8), (uint64_t) pow(2, 8), 3}
    };

    std::cout << "Third Test" << std::endl;
    double *signal = (double *) malloc(sizeof(double) * 256 * 10);
    int dim_arry[3] = {1, 1, 1};
    for (long long int i = 0; i < 256 * 10; i++)
        signal[i] = i % 100;
    cufftHandle plan;
    cufftDoubleComplex *data;
    data = static_cast<cufftDoubleComplex *>(malloc(sizeof(cufftDoubleComplex) * 256 * 10));


    /* Create a 1D FFT plan. */
    cufftPlanMany(&plan, 256, dim_arry, dim_arry, 1, 0,
                  dim_arry, 1, 0, CUFFT_Z2Z, 10);
    /* Use the CUFFT plan to transform the signal in place. */
    cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
    /* Destroy the CUFFT plan. */
    std::cout << data[0].x << data[0].y << std::endl;

    cufftDestroy(plan);
    cudaFree(data);

    cufftDoubleComplex *data1;
    data1 = static_cast<cufftDoubleComplex *>(malloc(sizeof(cufftDoubleComplex) * 256 * 10));

    for (long int i = 0; i < 256 * 10; i++) {
        data1[i].x = signal[i];
        data1[i].y = 0.;
    }

    vkfftPlanMany(plan_interface, 256, dim_arry, dim_arry, 1, 0,
                  dim_arry, 1, 0, CUFFT_Z2Z, 10, t);
    vkfftExecZ2Z(plan_interface, data1, data1, CUFFT_FORWARD, t);

    std::cout << data1[1].x << data1[1].y << std::endl;

    return 0;
}
