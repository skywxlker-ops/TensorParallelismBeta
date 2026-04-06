#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    std::cout << "Current path: " << std::filesystem::current_path() << std::endl;
    
    for (int i = 1; i <= 10; ++i) {
        std::string filename = "TP_MLP_Training_logs/TP_MLP_Training_log" + std::to_string(i) + ".csv";
        bool exists = std::filesystem::exists(filename);
        std::cout << "Checking " << filename << ": " << (exists ? "EXISTS" : "MISSING") << std::endl;
        
        if (exists) {
             // Check if we can open it
             std::ifstream f(filename);
             std::cout << "  -> Openable: " << (f.good() ? "YES" : "NO") << std::endl;
        }
    }
    return 0;
}
