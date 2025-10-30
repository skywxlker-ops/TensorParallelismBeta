#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <filesystem>

struct TensorMetadata {
    int version;
    std::vector<int> shape;
    std::string dtype;
    std::string timestamp;
};

struct Checkpoint {
    // === Timestamp helper ===
    static std::string timestamp() {
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
        return oss.str();
    }

    // === Determine next versioned checkpoint ===
    static std::string nextVersionedFile(const std::string& base_dir, int rank) {
        namespace fs = std::filesystem;
        int version = 1;
        while (fs::exists(base_dir + "/ckpt_rank_" + std::to_string(rank) + "_v" + std::to_string(version) + ".bin"))
            version++;
        return base_dir + "/ckpt_rank_" + std::to_string(rank) + "_v" + std::to_string(version) + ".bin";
    }

    // === Save tensor with binary + JSON metadata ===
    static bool save(const std::string& base_dir, const std::vector<float>& data, int rank) {
        namespace fs = std::filesystem;
        fs::create_directories(base_dir);

        std::string bin_file = nextVersionedFile(base_dir, rank);
        std::string json_file = bin_file.substr(0, bin_file.find_last_of('.')) + ".json";

        // Prepare metadata
        TensorMetadata meta;
        meta.version = extractVersion(bin_file);
        meta.shape = { static_cast<int>(data.size()) };
        meta.dtype = "float32";
        meta.timestamp = timestamp();

        // --- Write binary file (metadata + tensor) ---
        std::ofstream ofs(bin_file, std::ios::binary);
        if (!ofs) {
            std::cerr << "[Checkpoint] Rank " << rank << " failed to open " << bin_file << " for writing.\n";
            return false;
        }

        // Write metadata block
        int shape_size = meta.shape.size();
        ofs.write(reinterpret_cast<char*>(&meta.version), sizeof(int));
        ofs.write(reinterpret_cast<char*>(&shape_size), sizeof(int));
        ofs.write(reinterpret_cast<char*>(meta.shape.data()), shape_size * sizeof(int));

        int dtype_len = meta.dtype.size();
        ofs.write(reinterpret_cast<char*>(&dtype_len), sizeof(int));
        ofs.write(meta.dtype.c_str(), dtype_len);

        std::string ts = meta.timestamp;
        int ts_len = ts.size();
        ofs.write(reinterpret_cast<char*>(&ts_len), sizeof(int));
        ofs.write(ts.c_str(), ts_len);

        // Write tensor data
        ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        ofs.close();

        // --- Write human-readable metadata JSON ---
        std::ofstream meta_json(json_file);
        if (meta_json) {
            meta_json << "{\n";
            meta_json << "  \"rank\": " << rank << ",\n";
            meta_json << "  \"version\": " << meta.version << ",\n";
            meta_json << "  \"shape\": [" << data.size() << "],\n";
            meta_json << "  \"dtype\": \"" << meta.dtype << "\",\n";
            meta_json << "  \"tensor_name\": \"DTensorMain\",\n";
            meta_json << "  \"timestamp\": \"" << meta.timestamp << "\"\n";
            meta_json << "}\n";
        }

        std::cout << "[Checkpoint] Rank " << rank << " saved " << data.size()
                  << " elements to " << bin_file << " (version " << meta.version << ")\n";
        return true;
    }

    // === Load latest checkpoint with metadata ===
    static std::pair<std::vector<float>, TensorMetadata> loadLatest(const std::string& base_dir, int rank) {
        namespace fs = std::filesystem;
        std::vector<float> buffer;
        TensorMetadata meta{};

        // Find latest checkpoint
        int latest_version = 0;
        std::string latest_bin;
        for (auto& f : fs::directory_iterator(base_dir)) {
            std::string path = f.path().string();
            std::string match = "ckpt_rank_" + std::to_string(rank) + "_v";
            if (path.find(match) != std::string::npos && path.rfind(".bin") == path.size() - 4) {
                int v = extractVersion(path);
                if (v > latest_version) {
                    latest_version = v;
                    latest_bin = path;
                }
            }
        }

        if (latest_bin.empty()) {
            std::cerr << "[Checkpoint] Rank " << rank << " found no checkpoint in " << base_dir << "\n";
            return {{}, meta};
        }

        // Read metadata + data
        std::ifstream ifs(latest_bin, std::ios::binary);
        if (!ifs) {
            std::cerr << "[Checkpoint] Rank " << rank << " failed to open " << latest_bin << "\n";
            return {{}, meta};
        }

        int shape_size;
        ifs.read(reinterpret_cast<char*>(&meta.version), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&shape_size), sizeof(int));
        meta.shape.resize(shape_size);
        ifs.read(reinterpret_cast<char*>(meta.shape.data()), shape_size * sizeof(int));

        int dtype_len;
        ifs.read(reinterpret_cast<char*>(&dtype_len), sizeof(int));
        meta.dtype.resize(dtype_len);
        ifs.read(meta.dtype.data(), dtype_len);

        int ts_len;
        ifs.read(reinterpret_cast<char*>(&ts_len), sizeof(int));
        meta.timestamp.resize(ts_len);
        ifs.read(meta.timestamp.data(), ts_len);

        int numel = meta.shape[0];
        buffer.resize(numel);
        ifs.read(reinterpret_cast<char*>(buffer.data()), numel * sizeof(float));
        ifs.close();

        std::cout << "[Checkpoint] Rank " << rank << " loaded " << numel
                  << " elements from " << latest_bin
                  << " (version " << meta.version << ", dtype " << meta.dtype
                  << ", time " << meta.timestamp << ")\n";

        return {buffer, meta};
    }

private:
    static int extractVersion(const std::string& path) {
        size_t start = path.find("_v");
        size_t end = path.find(".bin");
        if (start == std::string::npos || end == std::string::npos) return 1;
        return std::stoi(path.substr(start + 2, end - start - 2));
    }
};
