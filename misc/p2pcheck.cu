#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    printf("device count: %d\n", n);
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("GPU %d: %s  sm_%d%d  asyncEngineCount=%d  "
               "concurrentKernels=%d  unifiedAddressing=%d\n",
               d, p.name, p.major, p.minor, p.asyncEngineCount,
               p.concurrentKernels, p.unifiedAddressing);
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) {
                int can = 0;
                cudaDeviceCanAccessPeer(&can, i, j);
                printf("canAccessPeer %d->%d : %d\n", i, j, can);
            }
    return 0;
}
