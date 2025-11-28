#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cassert>
#include <thread>
#include <chrono>

#include "process_group/error_handler.h"

using namespace dtensor;

// =============================================================================
// Test Utilities
// =============================================================================

void print_test_header(const std::string& test_name) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << test_name << "\n";
    std::cout << "========================================\n";
}

void print_result(const std::string& test_name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cout << "[FAIL] " << test_name << "\n";
    }
}

// =============================================================================
// Test 1: Error Classification
// =============================================================================

void test_error_classification() {
    print_test_header("Error Classification");
    
    // Test NCCL error classification
    assert(ErrorHandler::classifyNCCLError(ncclSuccess) == ErrorType::TRANSIENT);
    assert(ErrorHandler::classifyNCCLError(ncclInvalidArgument) == ErrorType::FATAL);
    assert(ErrorHandler::classifyNCCLError(ncclUnhandledCudaError) == ErrorType::TRANSIENT);
    std::cout << "  NCCL error classification [OK]\n";
    
    // Test CUDA error classification
    assert(ErrorHandler::classifyCUDAError(cudaSuccess) == ErrorType::TRANSIENT);
    assert(ErrorHandler::classifyCUDAError(cudaErrorInvalidValue) == ErrorType::FATAL);
    assert(ErrorHandler::classifyCUDAError(cudaErrorNotReady) == ErrorType::TRANSIENT);
    std::cout << "  CUDA error classification [OK]\n";
    
    print_result("Error Classification", true);
}

// =============================================================================
// Test 2: Exception Throwing
// =============================================================================

void test_exception_throwing() {
    print_test_header("Exception Throwing");
    
    bool caught_nccl = false;
    bool caught_cuda = false;
    
    // Test NCCL exception
    try {
        NCCL_CHECK_THROW(ncclInvalidArgument);
    } catch (const NCCLException& e) {
        caught_nccl = true;
        std::cout << "  Caught NCCL exception: " << e.what() << "\n";
        assert(e.getNCCLCode() == ncclInvalidArgument);
        assert(e.getType() == ErrorType::FATAL);
    }
    assert(caught_nccl);
    
    // Test CUDA exception
    try {
        CUDA_CHECK_THROW(cudaErrorInvalidValue);
    } catch (const CUDAException& e) {
        caught_cuda = true;
        std::cout << "  Caught CUDA exception: " << e.what() << "\n";
        assert(e.getCUDACode() == cudaErrorInvalidValue);
        assert(e.getType() == ErrorType::FATAL);
    }
    assert(caught_cuda);
    
    print_result("Exception Throwing", true);
}

// =============================================================================
// Test 3: Retry Mechanism
// =============================================================================

void test_retry_mechanism() {
    print_test_header("Retry Mechanism");
    
    // Simulate a flaky operation that fails first 2 times, then succeeds
    int attempt_count = 0;
    auto flaky_operation = [&attempt_count]() -> int {
        attempt_count++;
        if (attempt_count < 3) {
            // Simulate transient failure
            std::cout << "  Attempt " << attempt_count << ": simulating transient failure\n";
            ErrorDetails details;
            details.type = ErrorType::TRANSIENT;
            details.severity = ErrorSeverity::WARNING;
            details.message = "Simulated transient error";
            details.location = "test:retry";
            throw NCCLException(details);
        }
        std::cout << "  Attempt " << attempt_count << ": success!\n";
        return 42;
    };
    
    int result = ErrorHandler::executeWithRetry(
        flaky_operation,
        __FILE__,
        __LINE__,
        5,     // max retries
        50     // retry delay ms
    );
    
    assert(result == 42);
    assert(attempt_count == 3);  // Failed twice, succeeded on third
    std::cout << "  Retry succeeded after " << (attempt_count - 1) << " retries\n";
    
    print_result("Retry Mechanism", true);
}

// =============================================================================
// Test 4: Fatal Error No Retry
// =============================================================================

void test_fatal_no_retry() {
    print_test_header("Fatal Error No Retry");
    
    int attempt_count = 0;
    auto fatal_operation = [&attempt_count]() -> int {
        attempt_count++;
        std::cout << "  Attempt " << attempt_count << "\n";
        ErrorDetails details;
        details.type = ErrorType::FATAL;
        details.severity = ErrorSeverity::CRITICAL;
        details.message = "Simulated fatal error";
        details.location = "test:fatal";
        throw NCCLException(details);
    };
    
    bool caught = false;
    try {
        ErrorHandler::executeWithRetry(
            fatal_operation,
            __FILE__,
            __LINE__,
            5      // max retries (should not retry fatal errors)
        );
    } catch (const NCCLException& e) {
        caught = true;
        std::cout << "  Fatal error not retried: " << e.what() << "\n";
    }
    
    assert(caught);
    assert(attempt_count == 1);  // Should only try once for fatal errors
    std::cout << "  Fatal error correctly not retried\n";
    
    print_result("Fatal Error No Retry", true);
}

// =============================================================================
// Test 5: Max Retries Exceeded
// =============================================================================

void test_max_retries_exceeded() {
    print_test_header("Max Retries Exceeded");
    
    int attempt_count = 0;
    auto always_failing = [&attempt_count]() -> int {
        attempt_count++;
        ErrorDetails details;
        details.type = ErrorType::TRANSIENT;
        details.severity = ErrorSeverity::WARNING;
        details.message = "Always fails";
        details.location = "test:max_retries";
        throw NCCLException(details);
    };
    
    bool caught = false;
    try {
        ErrorHandler::executeWithRetry(
            always_failing,
            __FILE__,
            __LINE__,
            3,     // max 3 retries
            10     // fast retry for test
        );
    } catch (const NCCLException& e) {
        caught = true;
        std::cout << "  Max retries exceeded: " << e.what() << "\n";
    }
    
    assert(caught);
    assert(attempt_count == 4);  // Initial + 3 retries
    std::cout << "  Correctly stopped after max retries (" << (attempt_count - 1) << ")\n";
    
    print_result("Max Retries Exceeded", true);
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "===========================================\n";
    std::cout << "  Error Recovery & Fault Tolerance Tests\n";
    std::cout << "===========================================\n";
    
    try {
        // Run all tests
        test_error_classification();
        test_exception_throwing();
        test_retry_mechanism();
        test_fatal_no_retry();
        test_max_retries_exceeded();
        
        std::cout << "\n===========================================\n";
        std::cout << "  ALL TESTS PASSED\n";
        std::cout << "===========================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
