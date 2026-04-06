// ---------------------------------------------------------------------------
// CublasLtMatmul.cu
//
// cuBLASLt matmul with heuristic algorithm caching.
// See CublasLtMatmul.h for interface documentation.
// ---------------------------------------------------------------------------

#include "dnn/CublasLtMatmul.h"
#include "device/DeviceCore.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace OwnTensor {
namespace dnn {

// ── Singleton cuBLASLt handle ────────────────────────────────────────────────

static cublasLtHandle_t g_lt_handle = nullptr;
static std::mutex g_lt_mutex;

static cublasLtHandle_t get_lt_handle() {
    if (!g_lt_handle) {
        std::lock_guard<std::mutex> lock(g_lt_mutex);
        if (!g_lt_handle) {
            cublasStatus_t st = cublasLtCreate(&g_lt_handle);
            if (st != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "cublasLtCreate failed with status " + std::to_string(st));
            }
        }
    }
    return g_lt_handle;
}

// ── Algorithm cache ──────────────────────────────────────────────────────────
// Key: (M, K, N)  →  best cublasLtMatmulAlgo_t

struct ShapeKey {
    int M, K, N;
    bool operator==(const ShapeKey& o) const {
        return M == o.M && K == o.K && N == o.N;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const {
        // simple hash combine
        size_t h = std::hash<int>()(k.M);
        h ^= std::hash<int>()(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct CachedAlgo {
    cublasLtMatmulAlgo_t algo;
    size_t workspace_bytes;
};

static std::unordered_map<ShapeKey, CachedAlgo, ShapeKeyHash> g_algo_cache;
static std::mutex g_cache_mutex;

// Persistent workspace (grown as needed, never freed)
static void*  g_workspace     = nullptr;
static size_t g_workspace_cap = 0;

static void ensure_workspace(size_t needed) {
    if (needed <= g_workspace_cap) return;
    if (g_workspace) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, needed);
    g_workspace_cap = needed;
}

// ── Public API ───────────────────────────────────────────────────────────────

Tensor cublaslt_matmul(const Tensor& A, const Tensor& B) {
    // Validate inputs
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error(
            "cublaslt_matmul: A and B must be 2D [M,K] and [K,N]");
    }
    if (A.dtype() != Dtype::Float32 || B.dtype() != Dtype::Float32) {
        throw std::runtime_error("cublaslt_matmul: only Float32 supported");
    }

    const int M = static_cast<int>(A.shape().dims[0]);
    const int K = static_cast<int>(A.shape().dims[1]);
    const int N = static_cast<int>(B.shape().dims[1]);

    if (B.shape().dims[0] != K) {
        throw std::runtime_error(
            "cublaslt_matmul: inner dims mismatch A[M,K]@B[K,N]");
    }

    // Allocate output
    TensorOptions opts = A.opts().with_req_grad(false);
    Shape out_shape({{static_cast<int64_t>(M), static_cast<int64_t>(N)}});
    Tensor C = Tensor::empty(out_shape, opts);

    cublasLtHandle_t handle = get_lt_handle();
    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    // ── Create matrix descriptors ────────────────────────────────────────
    // cuBLASLt is column-major by default.
    // For row-major A[M,K]: treat as col-major A^T[K,M], so pass (K, M).
    // For row-major B[K,N]: treat as col-major B^T[N,K], so pass (N, K).
    // C = A @ B  =>  C^T = B^T @ A^T  (in col-major: C_cm = B_cm @ A_cm)
    //   B_cm is [N, K] with ld=N, A_cm is [K, M] with ld=K, C_cm is [N, M] with ld=N

    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUDA_R_32F);

    // A_cm = A^T => [K, M], ld = K
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K);
    // B_cm = B^T => [N, K], ld = N
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N);
    // C_cm = C^T => [N, M], ld = N
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N);

    // ── Lookup or find best algorithm ────────────────────────────────────
    ShapeKey key{M, K, N};
    CachedAlgo cached;

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_algo_cache.find(key);
        if (it != g_algo_cache.end()) {
            cached = it->second;
        } else {
            // Run heuristic search
            cublasLtMatmulPreference_t pref = nullptr;
            cublasLtMatmulPreferenceCreate(&pref);

            // Allow up to 32MB workspace
            size_t max_ws = 32ULL * 1024 * 1024;
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &max_ws, sizeof(max_ws));

            cublasLtMatmulHeuristicResult_t results[8];
            int returnedResults = 0;

            cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
                handle, matmulDesc,
                Bdesc, Adesc, Cdesc, Cdesc,  // B_cm @ A_cm = C_cm
                pref, 8, results, &returnedResults);

            cublasLtMatmulPreferenceDestroy(pref);

            if (st != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
                // Fallback: use default algorithm (no heuristic)
                // This shouldn't happen on modern GPUs with FP32
                cublasLtMatmulDescDestroy(matmulDesc);
                cublasLtMatrixLayoutDestroy(Adesc);
                cublasLtMatrixLayoutDestroy(Bdesc);
                cublasLtMatrixLayoutDestroy(Cdesc);

                // Fall back to basic cuBLAS gemm
                throw std::runtime_error(
                    "cublasLtMatmulAlgoGetHeuristic failed, no algorithms "
                    "found for shape (" + std::to_string(M) + ", " +
                    std::to_string(K) + ", " + std::to_string(N) + ")");
            }

            // Pick the first (best) result
            cached.algo = results[0].algo;
            cached.workspace_bytes = results[0].workspaceSize;
            g_algo_cache[key] = cached;
        }
    }

    // ── Ensure workspace ─────────────────────────────────────────────────
    ensure_workspace(cached.workspace_bytes);

    // ── Execute matmul ───────────────────────────────────────────────────
    float alpha = 1.0f, beta = 0.0f;

    // C_cm = B_cm @ A_cm  (col-major view of C = A @ B in row-major)
    cublasStatus_t st = cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        B.data<float>(), Bdesc,   // "A" in cuBLASLt = B_cm
        A.data<float>(), Adesc,   // "B" in cuBLASLt = A_cm
        &beta,
        C.data<float>(), Cdesc,   // C out
        C.data<float>(), Cdesc,   // D = C (in-place)
        &cached.algo,
        g_workspace, cached.workspace_bytes,
        stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cublasLtMatmul failed with status " + std::to_string(st));
    }

    // ── Cleanup descriptors ──────────────────────────────────────────────
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);

    return C;
}

} // namespace dnn
} // namespace OwnTensor
