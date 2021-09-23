cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(vkfft_execZ2Z CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(vkfft_execZ2Z main.cu)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35 60 70 75 80 86)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_CUDA_FLAGS "-g -O0")
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/vkFFT ${CMAKE_CURRENT_SOURCE_DIR}/VkFFT/benchmark_scripts/vkFFT_scripts/include)
add_library(VkFFT INTERFACE)

target_link_libraries(${PROJECT_NAME} PUBLIC VkFFT cuda ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB} fftw3 cufft)
