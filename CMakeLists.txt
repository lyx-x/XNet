CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(XNet)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include)

SET(CUDA_CUDNN_LIBRARY "/usr/local/cuda/lib64/libcudnn.so")

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FILE(GLOB SOURCES "src/*.cu" "src/*.cpp")
FILE(GLOB UTILS "src/utils/*.cu" "src/utils/*.cpp")
FILE(GLOB MODEL "src/model/*.cu" "src/model/*.cpp")
FILE(GLOB LAYER "src/layer/*.cu" "src/layer/*.cpp")
FILE(GLOB DATA "src/dataset/*.cu" "src/dataset/*.cpp")

CUDA_ADD_EXECUTABLE(XNet ${SOURCES} ${UTILS} ${MODEL} ${LAYER} ${DATA})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math ")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

TARGET_LINK_LIBRARIES(XNet ${OpenCV_LIBS} ${CUDA_CUDART_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUDNN_LIBRARY})