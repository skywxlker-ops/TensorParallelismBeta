#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <functional>

namespace dtensor {

// =============================================================================
// Error Classification
// =============================================================================

enum class ErrorType {
    TRANSIENT,      // Recoverable - network hiccup, temporary resource unavailable
    FATAL,          // Unrecoverable - invalid usage, hardware failure
    TIMEOUT         // Operation exceeded time limit
};

enum class ErrorSeverity {
    WARNING,        // Recoverable with retry
    ERROR,          // Requires intervention but not fatal
    CRITICAL        // Fatal, requires shutdown
};

// =============================================================================
// Error Details
// =============================================================================

struct ErrorDetails {
    ErrorType type;
    ErrorSeverity severity;
    std::string message;
    std::string location;      // __FILE__:__LINE__
    int retry_count;
    std::chrono::steady_clock::time_point timestamp;
    
    // NCCL specific
    ncclResult_t nccl_code;
    
    // CUDA specific  
    cudaError_t cuda_code;
    
    ErrorDetails()
        : type(ErrorType::FATAL),
          severity(ErrorSeverity::CRITICAL),
          retry_count(0),
          timestamp(std::chrono::steady_clock::now()),
          nccl_code(ncclSuccess),
          cuda_code(cudaSuccess) {}
};

// =============================================================================
// Exception Classes
// =============================================================================

class DTensorException : public std::runtime_error {
public:
    explicit DTensorException(const ErrorDetails& details)
        : std::runtime_error(details.message), details_(details) {}
    
    const ErrorDetails& getDetails() const { return details_; }
    ErrorType getType() const { return details_.type; }
    ErrorSeverity getSeverity() const { return details_.severity; }
    
protected:
    ErrorDetails details_;
};

class NCCLException : public DTensorException {
public:
    explicit NCCLException(const ErrorDetails& details)
        : DTensorException(details) {}
    
    ncclResult_t getNCCLCode() const { return details_.nccl_code; }
};

class CUDAException : public DTensorException {
public:
    explicit CUDAException(const ErrorDetails& details)
        : DTensorException(details) {}
    
    cudaError_t getCUDACode() const { return details_.cuda_code; }
};

class TimeoutException : public DTensorException {
public:
    explicit TimeoutException(const ErrorDetails& details)
        : DTensorException(details) {
        details_.type = ErrorType::TIMEOUT;
    }
};

// =============================================================================
// Error Handler
// =============================================================================

class ErrorHandler {
public:
    // NCCL error classification
    static ErrorType classifyNCCLError(ncclResult_t result);
    
    // CUDA error classification
    static ErrorType classifyCUDAError(cudaError_t error);
    
    // Execute function with retry on transient errors
    template<typename Func, typename... Args>
    static auto executeWithRetry(
        Func&& func,
        const char* file,
        int line,
        int max_retries = 3,
        int retry_delay_ms = 100
    ) -> decltype(func()) {
        
        int attempt = 0;
        while (attempt <= max_retries) {
            try {
                return func();
            } catch (const NCCLException& e) {
                if (e.getType() == ErrorType::TRANSIENT && attempt < max_retries) {
                    logRetry(e.getDetails(), attempt + 1, max_retries);
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(retry_delay_ms * (attempt + 1))
                    );
                    attempt++;
                    continue;
                }
                throw;  // Rethrow if not retryable or max retries exceeded
            } catch (const CUDAException& e) {
                if (e.getType() == ErrorType::TRANSIENT && attempt < max_retries) {
                    logRetry(e.getDetails(), attempt + 1, max_retries);
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(retry_delay_ms * (attempt + 1))
                    );
                    attempt++;
                    continue;
                }
                throw;
            }
        }
        
        // Should never reach here
       throw std::runtime_error("executeWithRetry: unexpected code path");
    }
    
    // Check NCCL result and throw on error
    static void checkNCCL(
        ncclResult_t result,
        const char* file,
        int line
    );
    
    // Check CUDA result and throw on error
    static void checkCUDA(
        cudaError_t error,
        const char* file,
        int line
    );
    
    // Log error details
    static void logError(const ErrorDetails& details);
    
    // Log retry attempt
    static void logRetry(const ErrorDetails& details, int attempt, int max_attempts);
    
    // Get human-readable error message
    static std::string formatError(const ErrorDetails& details);

private:
    // Retry configuration
    static constexpr int DEFAULT_MAX_RETRIES = 3;
    static constexpr int BASE_RETRY_DELAY_MS = 100;
};

// =============================================================================
// Enhanced Error Macros
// =============================================================================

// Macro for NCCL calls with automatic error handling
#define NCCL_CHECK_THROW(call) do { \
    ncclResult_t __result = (call); \
    dtensor::ErrorHandler::checkNCCL(__result, __FILE__, __LINE__); \
} while(0)

// Macro for CUDA calls with automatic error handling
#define CUDA_CHECK_THROW(call) do { \
    cudaError_t __error = (call); \
    dtensor::ErrorHandler::checkCUDA(__error, __FILE__, __LINE__); \
} while(0)

// Execute with retry (for NCCL operations)
#define NCCL_WITH_RETRY(call, max_retries) \
    dtensor::ErrorHandler::executeWithRetry( \
        [&]() -> ncclResult_t { \
            ncclResult_t __res = (call); \
            dtensor::ErrorHandler::checkNCCL(__res, __FILE__, __LINE__); \
            return __res; \
        }, \
        __FILE__, __LINE__, max_retries \
    )

} // namespace dtensor
