#include <iostream>
#include <vector>
#include <stdexcept>

// ============================================================================
// WITHOUT RAII - Manual Memory Management
// ============================================================================

void simple_array_without_raii() {
    std::cout << "\n=== WITHOUT RAII ===\n";
    
    int* numbers = new int[100];  // ❌ Manual allocation
    
    // Initialize
    for (int i = 0; i < 100; i++) {
        numbers[i] = i * 2;
    }
    
    std::cout << "First 5 elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << numbers[i] << " ";
    }
    std::cout << "\n";
    
    // Simulate error condition
    bool has_error = false;
    if (has_error) {
        delete[] numbers;  // ❌ Must remember to cleanup!
        throw std::runtime_error("Error occurred");
    }
    
    delete[] numbers;  // ❌ If exception thrown above, this leaks!
    std::cout << "Memory freed manually (if no exception)\n";
}

// ============================================================================
// WITH RAII - Automatic Memory Management
// ============================================================================

void simple_array_with_raii() {
    std::cout << "\n=== WITH RAII ===\n";
    
    std::vector<int> numbers(100);  // ✅ RAII: Auto allocation + cleanup
    
    // Initialize
    for (int i = 0; i < 100; i++) {
        numbers[i] = i * 2;
    }
    
    std::cout << "First 5 elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << numbers[i] << " ";
    }
    std::cout << "\n";
    
    // Simulate error condition
    bool has_error = false;
    if (has_error) {
        throw std::runtime_error("Error occurred");
        // ✅ Memory automatically freed by vector destructor!
    }
    
    // ✅ Memory automatically freed when vector goes out of scope
    std::cout << "Memory will be freed automatically\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "RAII Example 1: Simple Array\n";
    std::cout << "=============================\n";
    
    try {
        simple_array_without_raii();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception (without RAII): " << e.what() << "\n";
    }
    
    try {
        simple_array_with_raii();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception (with RAII): " << e.what() << "\n";
    }
    
    std::cout << "\nProgram completed successfully.\n";
    return 0;
}
