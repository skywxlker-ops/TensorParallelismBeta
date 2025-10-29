#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

struct Checkpoint {
    static bool save(const std::string& filename, const std::vector<float>& data, int rank) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cerr << "[Checkpoint] Rank " << rank << " failed to open " << filename << " for writing.\n";
            return false;
        }
        ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        ofs.close();
        std::cout << "[Checkpoint] Rank " << rank << " saved " << data.size() << " elements to " << filename << "\n";
        return true;
    }

    static std::vector<float> load(const std::string& filename, int rank) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            std::cerr << "[Checkpoint] Rank " << rank << " failed to open " << filename << " for reading.\n";
            return {};
        }
        ifs.seekg(0, std::ios::end);
        std::streamsize size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<float> buffer(size / sizeof(float));
        ifs.read(reinterpret_cast<char*>(buffer.data()), size);
        ifs.close();
        std::cout << "[Checkpoint] Rank " << rank << " loaded " << buffer.size() << " elements from " << filename << "\n";
        return buffer;
    }
};
