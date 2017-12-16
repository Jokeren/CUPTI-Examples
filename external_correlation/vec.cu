#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <atomic>
#include <string>

#define DRIVER_API_CALL(apiFuncCall)                                           \
  do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
        __FILE__, __LINE__, #apiFuncCall, _status);                    \
      exit(-1);                                                              \
    }                                                                          \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
  do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
        __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
      exit(-1);                                                              \
    }                                                                          \
  } while (0)

#define COMPUTE_N 50000

extern void initTrace(void);
extern void finiTrace(void);

extern __thread int64_t localId;

static std::atomic<int> correlationId;


static __global__ void
vecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}


static __global__ void
vecSub(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] - B[i];
}


static void
do_pass_runtime(CUdevice device)
{
  int *h_A, *h_B, *h_C;
  int *d_A1, *d_B1, *d_C1;
  int *d_A2, *d_B2, *d_C2;
  size_t size = COMPUTE_N * sizeof(int);
  int threadsPerBlock = 256;
  int blocksPerGrid = 0;

  // Allocate input vectors h_A and h_B in host memory
  // don't bother to initialize
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);

  // Allocate vectors in device memory
  RUNTIME_API_CALL(cudaMalloc((void**)&d_A1, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B1, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C1, size));
  RUNTIME_API_CALL(cudaMemcpy(d_A1, h_A, size, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(d_B1, h_B, size, cudaMemcpyHostToDevice));

  RUNTIME_API_CALL(cudaMalloc((void**)&d_A2, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B2, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C2, size));
  RUNTIME_API_CALL(cudaMemcpy(d_A2, h_A, size, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice));

  blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  localId = correlationId.fetch_add(1);
  printf("Run VecAdd\n");
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A1, d_B1, d_C1, COMPUTE_N);

  localId = correlationId.fetch_add(1);
  printf("Run VecSub\n");
  vecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A2, d_B2, d_C2, COMPUTE_N);

  // Clean up
  RUNTIME_API_CALL(cudaFree(d_A1));
  RUNTIME_API_CALL(cudaFree(d_B1));
  RUNTIME_API_CALL(cudaFree(d_C1));
  RUNTIME_API_CALL(cudaFree(d_A2));
  RUNTIME_API_CALL(cudaFree(d_B2));
  RUNTIME_API_CALL(cudaFree(d_C2));

  free(h_A);
  free(h_B);
  free(h_C);
}


static void
do_pass_same_context(CUdevice device)
{
  CUcontext deviceContext;
  cuCtxCreate(&deviceContext, 0, device);
  CUpti_ActivityPCSamplingConfig configPC;
  configPC.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
  cuptiActivityConfigurePCSampling(deviceContext, &configPC);

  CUmodule moduleAdd;
  CUfunction vecAdd;
  DRIVER_API_CALL(cuModuleLoad(&moduleAdd, "vecAdd.cubin"));
  DRIVER_API_CALL(cuModuleGetFunction(&vecAdd, moduleAdd, "vecAdd"));

  CUmodule moduleSub;
  CUfunction vecSub;
  DRIVER_API_CALL(cuModuleLoad(&moduleSub, "vecSub.cubin"));
  DRIVER_API_CALL(cuModuleGetFunction(&vecSub, moduleSub, "vecSub"));

#pragma omp parallel 
  {
    CUcontext context;
    cuCtxSetCurrent(deviceContext);
    cuCtxGetCurrent(&context);

    int *h_A, *h_B, *h_C;
    int *d_A1, *d_B1, *d_C1;
    int *d_A2, *d_B2, *d_C2;
    size_t size = COMPUTE_N * sizeof(int);
    int threadsPerBlock = 256;
    int blocksPerGrid = 0;

    // Allocate input vectors h_A and h_B in host memory
    // don't bother to initialize
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
    int computeN = COMPUTE_N;

    if (omp_get_thread_num() == 0) {
      // Allocate vectors in device memory
      RUNTIME_API_CALL(cudaMalloc((void**)&d_A1, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_B1, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_C1, size));
      RUNTIME_API_CALL(cudaMemcpy(d_A1, h_A, size, cudaMemcpyHostToDevice));
      RUNTIME_API_CALL(cudaMemcpy(d_B1, h_B, size, cudaMemcpyHostToDevice));

      // Launch kernel
      localId = correlationId.fetch_add(1);
      printf("Run VecAdd\n");
      void *args1[4] = {
        &d_A1, &d_B1, &d_C1, &computeN
      };
      DRIVER_API_CALL(cuLaunchKernel(vecAdd, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args1, 0));

      // Clean up
      RUNTIME_API_CALL(cudaFree(d_A1));
      RUNTIME_API_CALL(cudaFree(d_B1));
      RUNTIME_API_CALL(cudaFree(d_C1));
    } else if (omp_get_thread_num() == 1) {
      // Allocate vectors in device memory
      RUNTIME_API_CALL(cudaMalloc((void**)&d_A2, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_B2, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_C2, size));
      RUNTIME_API_CALL(cudaMemcpy(d_A2, h_A, size, cudaMemcpyHostToDevice));
      RUNTIME_API_CALL(cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice));

      // Launch kernel
      localId = correlationId.fetch_add(1);
      printf("Run VecSub\n");
      void *args2[4] = {
        &d_A2, &d_B2, &d_C2, &computeN
      };
      DRIVER_API_CALL(cuLaunchKernel(vecSub, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args2, 0));

      // Clean up
      RUNTIME_API_CALL(cudaFree(d_A2));
      RUNTIME_API_CALL(cudaFree(d_B2));
      RUNTIME_API_CALL(cudaFree(d_C2));
    }
    free(h_A);
    free(h_B);
    free(h_C);
  }
}


static void
do_pass_diff_context(CUdevice device)
{
#pragma omp parallel 
  {
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    CUpti_ActivityPCSamplingConfig configPC;
    configPC.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
    cuptiActivityConfigurePCSampling(context, &configPC);

    int *h_A, *h_B, *h_C;
    int *d_A1, *d_B1, *d_C1;
    int *d_A2, *d_B2, *d_C2;
    size_t size = COMPUTE_N * sizeof(int);
    int threadsPerBlock = 256;
    int blocksPerGrid = 0;

    // Allocate input vectors h_A and h_B in host memory
    // don't bother to initialize
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
    int computeN = COMPUTE_N;

    if (omp_get_thread_num() == 0) {
      CUmodule moduleAdd;
      CUfunction vecAdd;
      DRIVER_API_CALL(cuModuleLoad(&moduleAdd, "vecAdd.cubin"));
      DRIVER_API_CALL(cuModuleGetFunction(&vecAdd, moduleAdd, "vecAdd"));

      // Allocate vectors in device memory
      RUNTIME_API_CALL(cudaMalloc((void**)&d_A1, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_B1, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_C1, size));
      RUNTIME_API_CALL(cudaMemcpy(d_A1, h_A, size, cudaMemcpyHostToDevice));
      RUNTIME_API_CALL(cudaMemcpy(d_B1, h_B, size, cudaMemcpyHostToDevice));

      // Launch kernel
      localId = correlationId.fetch_add(1);
      printf("Run VecAdd\n");
      void *args1[4] = {
        &d_A1, &d_B1, &d_C1, &computeN
      };
      DRIVER_API_CALL(cuLaunchKernel(vecAdd, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args1, 0));

      // Clean up
      RUNTIME_API_CALL(cudaFree(d_A1));
      RUNTIME_API_CALL(cudaFree(d_B1));
      RUNTIME_API_CALL(cudaFree(d_C1));
    } else if (omp_get_thread_num() == 1) {
      CUmodule moduleSub;
      CUfunction vecSub;
      DRIVER_API_CALL(cuModuleLoad(&moduleSub, "vecSub.cubin"));
      DRIVER_API_CALL(cuModuleGetFunction(&vecSub, moduleSub, "vecSub"));

      // Allocate vectors in device memory
      RUNTIME_API_CALL(cudaMalloc((void**)&d_A2, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_B2, size));
      RUNTIME_API_CALL(cudaMalloc((void**)&d_C2, size));
      RUNTIME_API_CALL(cudaMemcpy(d_A2, h_A, size, cudaMemcpyHostToDevice));
      RUNTIME_API_CALL(cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice));

      // Launch kernel
      localId = correlationId.fetch_add(1);
      printf("Run VecSub\n");
      void *args2[4] = {
        &d_A2, &d_B2, &d_C2, &computeN
      };
      DRIVER_API_CALL(cuLaunchKernel(vecSub, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args2, 0));

      // Clean up
      RUNTIME_API_CALL(cudaFree(d_A2));
      RUNTIME_API_CALL(cudaFree(d_B2));
      RUNTIME_API_CALL(cudaFree(d_C2));
    }
    free(h_A);
    free(h_B);
    free(h_C);
  }
}


int
main(int argc, char *argv[])
{
  if (argc < 2) {
    exit(-1);
  }
  std::string option = std::string(argv[1]);

  CUdevice device;  
  char deviceName[32];
  int deviceNum = 0, devCount = 0;

  // initialize the activity trace
  // make sure activity is enabled before any CUDA API
  initTrace();

  DRIVER_API_CALL(cuInit(0));

  RUNTIME_API_CALL(cudaGetDeviceCount(&devCount));
  for (deviceNum=0; deviceNum<devCount; deviceNum++) {
    DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
    printf("Device Name: %s\n", deviceName);

    RUNTIME_API_CALL(cudaSetDevice(deviceNum));

    if (option == "runtime") {
      do_pass_runtime(device);
    } else if (option == "same_context") {
      do_pass_same_context(device);
    } else if (option == "diff_context") {
      do_pass_diff_context(device);
    } else {
      exit(-1);
    }

    RUNTIME_API_CALL(cudaDeviceSynchronize());
    RUNTIME_API_CALL(cudaDeviceReset());

    finiTrace();
  }
  return 0;
}
