#include "error_handler.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace dtensor {

// =============================================================================
// Error Classification
// =============================================================================

ErrorType ErrorHandler::classifyNCCLError(ncclResult_t result) {
    switch (result) {
        case ncclSuccess:
            return ErrorType::TRANSIENT;  // Not really an error
            
        // Transient errors - potentially recoverable
        case ncclUnhandledCudaError:
        case ncclSystemError:
            return ErrorType::TRANSIENT;
            
        // Fatal errors - programming bugs or hardware failures
        case ncclInvalidArgument:
        case ncclInvalidUsage:
        case ncclRemoteError:
        case ncclInternalError:
            return ErrorType::FATAL;
            
        // Default to fatal for unknown errors
        default:
            return ErrorType::FATAL;
    }
}

ErrorType ErrorHandler::classifyCUDAError(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return ErrorType::TRANSIENT;  // Not really an error
            
        // Transient errors
        case cudaErrorNotReady:
        case cudaErrorLaunchTimeout:
            return ErrorType::TRANSIENT;
            
        // Fatal errors
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidMemcpyDirection:
        case cudaErrorIllegalAddress:
        case cudaErrorNoDevice:
            return ErrorType::FATAL;
            
        // Default to transient for unknown errors (safer for retry)
        default:
            return ErrorType::TRANSIENT;
    }
}

// =============================================================================
// Error Checking
// =============================================================================

void ErrorHandler::checkNCCL(ncclResult_t result, const char* file, int line) {
    if (result == ncclSuccess) {
        return;
    }
    
    ErrorDetails details;
    details.nccl_code = result;
    details.type = classifyNCCLError(result);
    details.severity = (details.type == ErrorType::FATAL) 
                       ? ErrorSeverity::CRITICAL 
                       : ErrorSeverity::WARNING;
    
    std::ostringstream oss;
    oss << file << ":" << line;
    details.location = oss.str();
    
    std::ostringstream msg;
    msg << "NCCL Error: " << ncclGetErrorString(result)
        << " (" << result << ")";
    details.message = msg.str();
    
    logError(details);
    throw NCCLException(details);
}

void ErrorHandler::checkCUDA(cudaError_t error, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    
    ErrorDetails details;
    details.cuda_code = error;
    details.type = classifyCUDAError(error);
    details.severity = (details.type == ErrorType::FATAL)
                       ? ErrorSeverity::CRITICAL
                       : ErrorSeverity::WARNING;
    
    std::ostringstream oss;
    oss << file << ":" << line;
    details.location = oss.str();
    
    std::ostringstream msg;
    msg << "CUDA Error: " << cudaGetErrorString(error)
        << " (" << error << ")";
    details.message = msg.str();
    
    logError(details);
    throw CUDAException(details);
}

// =============================================================================
// Logging
// =============================================================================

void ErrorHandler::logError(const ErrorDetails& details) {
    std::cerr << "[ERROR] " << formatError(details) << std::endl;
}

void ErrorHandler::logRetry(const ErrorDetails& details, int attempt, int max_attempts) {
    std::cerr << "[RETRY " << attempt << "/" << max_attempts << "] " 
              << details.message << " at " << details.location << std::endl;
}

std::string ErrorHandler::formatError(const ErrorDetails& details) {
    std::ostringstream oss;
    
    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    oss << "[" << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << "] ";
    
    // Severity
    switch (details.severity) {
        case ErrorSeverity::WARNING:
            oss << "[WARN] ";
            break;
        case ErrorSeverity::ERROR:
            oss << "[ERROR] ";
            break;
        case ErrorSeverity::CRITICAL:
            oss << "[CRITICAL] ";
            break;
    }
    
    // Type
    switch (details.type) {
        case ErrorType::TRANSIENT:
            oss << "[TRANSIENT] ";
            break;
        case ErrorType::FATAL:
            oss << "[FATAL] ";
            break;
        case ErrorType::TIMEOUT:
            oss << "[TIMEOUT] ";
            break;
    }
    
    // Message and location
    oss << details.message << " at " << details.location;
    
    // Retry count if applicable
    if (details.retry_count > 0) {
        oss << " (retry " << details.retry_count << ")";
    }
    
    return oss.str();
}

} // namespace dtensor
