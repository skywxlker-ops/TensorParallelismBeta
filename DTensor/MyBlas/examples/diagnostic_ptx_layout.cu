// diagnostic_ptx_layout_v2.cu - Better diagnostic with identity matrix
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

// Diagnostic kernel with better initialization
__global__ void diagnostic_mma_layout_v2() {
    __shared__ __half As[16 * 16];
    __shared__ __half Bs[16 * 8];
    
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    
    // Initialize A as identity-like pattern: A[i][j] = i*16 + j
    for (int i = tid; i < 16 * 16; i += blockDim.x) {
        int row = i / 16;
        int col = i % 16;
        As[i] = __float2half((float)(row * 16 + col));
    }
    
    // Initialize B as ones
    for (int i = tid; i < 16 * 8; i += blockDim.x) {
        Bs[i] = __float2half(1.0f);
    }
    __syncthreads();
    
    // Each thread will perform mma independently to see the pattern
    uint32_t frag_a[4];
    uint32_t frag_b[2];
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float frag_d[4];
    
    // All threads load from the same location to see what each gets
    uint32_t smem_addr_a = __cvta_generic_to_shared(As);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
        : "r"(smem_addr_a)
    );
    
    uint32_t smem_addr_b = __cvta_generic_to_shared(Bs);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(frag_b[0]), "=r"(frag_b[1])
        : "r"(smem_addr_b)
    );
    
    // Perform mma.sync
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(frag_d[0]), "=f"(frag_d[1]), "=f"(frag_d[2]), "=f"(frag_d[3])
        : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
          "r"(frag_b[0]), "r"(frag_b[1]),
          "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[2]), "f"(frag_c[3])
    );
    
    // Print what each thread computed
    if (tid < 32) {
        printf("Lane %2d: [%.0f, %.0f, %.0f, %.0f]\n",
               lane_id, frag_d[0], frag_d[1], frag_d[2], frag_d[3]);
    }
    
    // Now let's try to understand the mapping by storing to different locations
    __shared__ float output_by_thread[32][4];
    if (tid < 32) {
        output_by_thread[lane_id][0] = frag_d[0];
        output_by_thread[lane_id][1] = frag_d[1];
        output_by_thread[lane_id][2] = frag_d[2];
        output_by_thread[lane_id][3] = frag_d[3];
    }
    __syncthreads();
    
    // Print organized view
    if (tid == 0) {
        printf("\n=== Organized by thread ===\n");
        for (int t = 0; t < 32; t++) {
            printf("T%02d: [%6.0f, %6.0f, %6.0f, %6.0f]\n", t,
                   output_by_thread[t][0], output_by_thread[t][1],
                   output_by_thread[t][2], output_by_thread[t][3]);
        }
        
        // Try to reconstruct the 16x8 matrix
        printf("\n=== Attempting to reconstruct 16x8 output ===\n");
        printf("Expected: each row should sum to (row*16 + 0) + (row*16 + 1) + ... + (row*16 + 15)\n");
        printf("Which equals: row*16*16 + (0+1+...+15) = row*256 + 120\n\n");
        
        for (int r = 0; r < 16; r++) {
            printf("Row %2d expected sum: %.0f\n", r, (float)(r * 256 + 120));
        }
    }
}

int main() {
    printf("=== PTX mma.sync Layout Diagnostic V2 ===\n\n");
    
    diagnostic_mma_layout_v2<<<1, 128>>>();
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}
