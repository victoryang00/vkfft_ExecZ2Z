#include <iostream>
#include <vkFFT.h>
#include <utils_VkFFT.h>
#include <cufftw.h>
#include <chrono>


VkFFTResult
performVulkanFFT(VkGPU *vkGPU, VkFFTApplication *app, VkFFTLaunchParams *launchParams, int inverse, uint64_t num_iter) {
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cudaError_t res = cudaSuccess;
    std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_iter; i++) {
        resFFT = VkFFTAppend(app, inverse, launchParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
    }
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
    std::cout << totTime << std::endl;
    return resFFT;
}
VkFFTResult performVulkanFFTiFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, uint64_t num_iter) {
    VkFFTResult resFFT = VKFFT_SUCCESS;

    cudaError_t res = cudaSuccess;
    std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_iter; i++) {
        resFFT = VkFFTAppend(app, -1, launchParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
        resFFT = VkFFTAppend(app, 1, launchParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
    }
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
    std::cout << totTime << std::endl;

    return resFFT;
}

VkFFTResult get_VkFFT_double(VkGPU *vkGPU) {
    VkFFTResult resFFT = VKFFT_SUCCESS;

    cudaError_t res = cudaSuccess;

    printf("12 - VkFFT/FFTW C2C precision test in double precision\n");
    const int num_benchmark_samples = 1;
    const int num_runs = 1;

    uint64_t benchmark_dimensions[num_benchmark_samples][4] = {
            {(uint64_t) pow(2, 8), (uint64_t) pow(2, 8), (uint64_t) pow(2, 8), 3}
    };

    for (int n = 0; n < num_benchmark_samples; n++) {
        for (int r = 0; r < num_runs; r++) {
            fftw_complex *inputC;
            fftw_complex *inputC_double;
            uint64_t dims[3] = {benchmark_dimensions[n][0], benchmark_dimensions[n][1], benchmark_dimensions[n][2]};

            inputC = (fftw_complex *) (malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
            if (!inputC) return VKFFT_ERROR_MALLOC_FAILED;
            inputC_double = (fftw_complex *) (malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
            if (!inputC_double) return VKFFT_ERROR_MALLOC_FAILED;
            for (uint64_t l = 0; l < dims[2]; l++) {
                for (uint64_t j = 0; j < dims[1]; j++) {
                    for (uint64_t i = 0; i < dims[0]; i++) {
                        inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double) (
                                2 * ((double) rand()) / RAND_MAX - 1.0);
                        inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double) (
                                2 * ((double) rand()) / RAND_MAX - 1.0);
                        inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (double) inputC[i + j * dims[0] +
                                                                                                    l * dims[0] *
                                                                                                    dims[1]][0];
                        inputC_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (double) inputC[i + j * dims[0] +
                                                                                                    l * dims[0] *
                                                                                                    dims[1]][1];
                    }
                }
            }

            fftw_plan p;

            fftw_complex *output_FFTW = (fftw_complex *) (malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
            if (!output_FFTW) return VKFFT_ERROR_MALLOC_FAILED;
            switch (benchmark_dimensions[n][3]) {
                case 1:
                    p = fftw_plan_dft_1d((int) benchmark_dimensions[n][0], inputC_double, output_FFTW, -1,
                                         FFTW_ESTIMATE);
                    break;
                case 2:
                    p = fftw_plan_dft_2d((int) benchmark_dimensions[n][1], (int) benchmark_dimensions[n][0],
                                         inputC_double, output_FFTW, -1, FFTW_ESTIMATE);
                    break;
                case 3:
                    p = fftw_plan_dft_3d((int) benchmark_dimensions[n][2], (int) benchmark_dimensions[n][1],
                                         (int) benchmark_dimensions[n][0], inputC_double, output_FFTW, -1,
                                         FFTW_ESTIMATE);
                    break;
            }

            fftw_execute(p);

            float totTime = 0;
            int num_iter = 1;

            //VkFFT part

            VkFFTConfiguration configuration = {};
            VkFFTApplication app = {};

            configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
            configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.
            configuration.size[1] = benchmark_dimensions[n][1];
            configuration.size[2] = benchmark_dimensions[n][2];

            //After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
            configuration.device = &vkGPU->device;
            configuration.doublePrecision = true;

            uint64_t numBuf = 1;

            //Allocate buffers for the input data. - we use 4 in this example
            uint64_t *bufferSize = (uint64_t *) malloc(sizeof(uint64_t) * numBuf);
            if (!bufferSize) return VKFFT_ERROR_MALLOC_FAILED;
            for (uint64_t i = 0; i < numBuf; i++) {
                bufferSize[i] = {};
                bufferSize[i] = (uint64_t) sizeof(double) * 2 * configuration.size[0] * configuration.size[1] *
                                configuration.size[2] / numBuf;
            }

            cuFloatComplex *buffer = 0;

            for (uint64_t i = 0; i < numBuf; i++) {

                res = cudaMalloc((void **) &buffer, bufferSize[i]);
                if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;

            }

            configuration.bufferNum = numBuf;
            // Can specify buffers at launch
            configuration.bufferSize = bufferSize;

            //Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
            uint64_t shift = 0;
            for (uint64_t i = 0; i < numBuf; i++) {

                res = cudaMemcpy(buffer, inputC, bufferSize[i], cudaMemcpyHostToDevice);
                if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;

                shift += bufferSize[i];
            }
            //Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.
            resFFT = initializeVkFFT(&app, configuration);
            if (resFFT != VKFFT_SUCCESS) return resFFT;
            //Submit FFT+iFFT.
            //num_iter = 1;
            //specify buffers at launch
            VkFFTLaunchParams launchParams = {};

            launchParams.buffer = (void **) &buffer;

            resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, num_iter);
            if (resFFT != VKFFT_SUCCESS) return resFFT;
            fftw_complex *output_VkFFT = (fftw_complex *) (malloc(sizeof(fftw_complex) * dims[0] * dims[1] * dims[2]));
            if (!output_VkFFT) return VKFFT_ERROR_MALLOC_FAILED;
            //Transfer data from GPU using staging buffer.
            shift = 0;
            for (uint64_t i = 0; i < numBuf; i++) {

                res = cudaMemcpy(output_VkFFT, buffer, bufferSize[i], cudaMemcpyDeviceToHost);
                if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;

                shift += bufferSize[i];
            }
            double avg_difference[2] = {0, 0};
            double max_difference[2] = {0, 0};
            double avg_eps[2] = {0, 0};
            double max_eps[2] = {0, 0};
            for (uint64_t l = 0; l < dims[2]; l++) {
                for (uint64_t j = 0; j < dims[1]; j++) {
                    for (uint64_t i = 0; i < dims[0]; i++) {
                        uint64_t loc_i = i;
                        uint64_t loc_j = j;
                        uint64_t loc_l = l;

                        double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] *
                                                        output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] +
                                                        output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] *
                                                        output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);

                        double current_diff_x_VkFFT = (
                                output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] -
                                output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
                        double current_diff_y_VkFFT = (
                                output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] -
                                output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
                        double current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT +
                                                              current_diff_y_VkFFT * current_diff_y_VkFFT);
                        if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
                        avg_difference[1] += current_diff_norm_VkFFT;
                        if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1]) && (current_data_norm > 1e-16)) {
                            max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
                        }
                        avg_eps[1] += (current_data_norm > 1e-10) ? current_diff_norm_VkFFT / current_data_norm : 0;
                    }
                }
            }
            avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
            avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
            avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
            avg_eps[1] /= (dims[0] * dims[1] * dims[2]);
            printf("VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.15f max_difference: %.15f avg_eps: %.15f max_eps: %.15f\n",
                   dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
            free(output_VkFFT);
            for (uint64_t i = 0; i < numBuf; i++) {
                cudaFree(buffer);
            }
#if defined(USE_cuFFT) || defined(USE_rocFFT)
            free(output_extFFT);
#endif
            deleteVkFFT(&app);
            free(inputC);
            fftw_destroy_plan(p);
            free(inputC_double);
            free(output_FFTW);
        }
    }
    return resFFT;
}

//vkfftPlanMany
VkFFTResult
vkfftPlanMany(VkGPU *vkGPU, VkFFTConfiguration configuration, VkFFTApplication appZ2Z, int rank, int *doubleComplex,
              int *doubleComplexPadded, int istride,
              int doubleComplexPaddedTotal, int *complexGridSizePadded, int ostride,
              int complexGridSizePaddedTotal, cufftType type, int batch, cudaStream_t *stream) {
    cuDoubleComplex *buffer = 0;
    cuDoubleComplex *buffer1 = 0;

    const int ZZ = 3, XX = 1, YY = 2;
    configuration.FFTdim = 3;
    configuration.size[0] = doubleComplex[ZZ];
    configuration.size[1] = doubleComplex[XX];
    configuration.size[2] = doubleComplex[YY];
    configuration.doublePrecision = true;
    //configuration.disableMergeSequencesR2C = 1;
    configuration.device = (CUdevice *) malloc(sizeof(CUdevice));
    cudaError_t result = cudaGetDevice(configuration.device);
    if (result != cudaSuccess) {
        printf("VKFFT_ERROR_FAILED_TO_GET_DEVICE error: %d\n", result);
        return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    }
    configuration.num_streams = 1;
    configuration.stream = stream;

    uint64_t bufferSize =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ] *
            sizeof(cufftDoubleComplex);
    std::cout << "bufferSize = " << bufferSize << std::endl;
    configuration.bufferSize = &bufferSize;
    configuration.bufferStride[0] = complexGridSizePadded[ZZ];
    configuration.bufferStride[1] = complexGridSizePadded[ZZ] * complexGridSizePadded[YY];
    configuration.bufferStride[2] = complexGridSizePadded[ZZ] * complexGridSizePadded[YY] * complexGridSizePadded[XX];
    result = cudaMalloc((void **) &buffer, bufferSize);
    if (result != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = (void **) &buffer;


    configuration.isInputFormatted = 1;
    configuration.inverseReturnToInputBuffer = 1;
    uint64_t inputBufferSize =
            doubleComplexPadded[XX] * doubleComplexPadded[YY] * doubleComplexPadded[ZZ] * sizeof(cufftDoubleComplex);
    std::cout << "inputBufferSize = " << inputBufferSize << std::endl;
    configuration.inputBufferSize = &inputBufferSize;
    configuration.inputBufferStride[0] = doubleComplexPadded[ZZ];
    configuration.inputBufferStride[1] = doubleComplexPadded[ZZ] * doubleComplexPadded[YY];
    configuration.inputBufferStride[2] = doubleComplexPadded[ZZ] * doubleComplexPadded[YY] * doubleComplexPadded[XX];
    result = cudaMemcpy(buffer, buffer1, inputBufferSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) return VKFFT_ERROR_FAILED_TO_COPY;
    configuration.inputBuffer = (void **) &buffer1;
    VkFFTResult resFFT = initializeVkFFT(&appZ2Z, configuration);
    if (resFFT != VKFFT_SUCCESS) printf("VkFFT error: %d\n", resFFT);


    std::cout << "vkFFT: complex dim = " << doubleComplexPadded[XX] << "x" << doubleComplexPadded[YY] << "x"
              << doubleComplexPadded[ZZ] << std::endl;
    return resFFT;
}

VkFFTResult vkfftExecZ2Z(VkGPU *vkGPU, VkFFTApplication appZ2Z, VkFFTConfiguration configuration,
                         cufftDoubleComplex *idata,
                         cufftDoubleComplex *odata,
                         int direction, cudaStream_t *stream) {
    uint64_t num_iter = (((uint64_t)4096 * 1024.0 * 1024.0) / *configuration.bufferSize > 1000) ? 1000 : (uint64_t)((uint64_t)4096 * 1024.0 * 1024.0) / *configuration.bufferSize;
    VkFFTLaunchParams launchParams={};
    auto resFFT = performVulkanFFTiFFT(vkGPU, &appZ2Z, &launchParams,num_iter);
    return resFFT;
}

int main() {
    VkGPU vkGPU = {};
    cufftHandle planR2C_;
    cudaStream_t *t;
    cudaStreamCreate(t);

    CUresult res = CUDA_SUCCESS;
    cudaError_t res2 = cudaSuccess;
    std::cout << "First Test" << std::endl;

    res = cuInit(0);
    if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    res2 = cudaSetDevice((int) vkGPU.device_id);
    if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
    res = cuDeviceGet(&vkGPU.device, (int) vkGPU.device_id);
    if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    res = cuCtxCreate(&vkGPU.context, 0, (int) vkGPU.device);
    if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
    get_VkFFT_double(&vkGPU);

    std::cout << "Second Test" << std::endl;
    cufftHandle plan;

    cufftComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftComplex)*256*10);
    /* Create a 1D FFT plan. */
    cufftPlan1d(&plan, 256, CUFFT_Z2Z, 10);
    /* Use the CUFFT plan to transform the signal in place. */
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    /* Destroy the CUFFT plan. */
    cufftDestroy(plan);
    cudaFree(data);
    return 0;
}
