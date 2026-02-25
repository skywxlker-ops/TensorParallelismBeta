#pragma once

#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "core/Serialization.h"
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <regex>
#include <chrono>
#include "checkpointing/RNG.h"

namespace OwnTensor {

namespace fs = std::filesystem;

/**
 * Saves a full training checkpoint.
 */
inline void save_checkpoint(const std::string& path, 
                           nn::Module& model, 
                           nn::Optimizer& optimizer, 
                           int epoch, 
                           float loss) {
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for writing: " + path);
    }

    // 1. Header
    os.write("CKPT", 4);
    int version = 1;
    os.write(reinterpret_cast<const char*>(&version), sizeof(int));

    // 2. Training state
    os.write(reinterpret_cast<const char*>(&epoch), sizeof(int));
    os.write(reinterpret_cast<const char*>(&loss), sizeof(float));

    // 3. Model state
    auto params = model.parameters();
    int count = static_cast<int>(params.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(int));
    for (const auto& p : params) {
        save_tensor(p, os);
    }

    // 4. Optimizer state
    optimizer.save_state(os);

    RNGState rng_state = RNG::get_state();
    os.write(reinterpret_cast<const char*>(&rng_state), sizeof(RNGState));

    os.close();
}

/**
 * Loads a full training checkpoint.
 */
inline void load_checkpoint(const std::string& path, 
                           nn::Module& model, 
                           nn::Optimizer& optimizer, 
                           int& epoch, 
                           float& loss) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for reading: " + path);
    }

    // 1. Header
    char magic[4];
    is.read(magic, 4);
    if (std::string(magic, 4) != "CKPT") {
        throw std::runtime_error("Invalid checkpoint format: " + path);
    }
    int version;
    is.read(reinterpret_cast<char*>(&version), sizeof(int));

    // 2. Training state
    is.read(reinterpret_cast<char*>(&epoch), sizeof(int));
    is.read(reinterpret_cast<char*>(&loss), sizeof(float));

    // 3. Model state
    auto params = model.parameters();
    int count;
    is.read(reinterpret_cast<char*>(&count), sizeof(int));
    if (count != static_cast<int>(params.size())) {
        throw std::runtime_error("Checkpoint model parameter count mismatch");
    }
    for (auto& p : params) {
        Tensor loaded = load_tensor(is);
        p.copy_(loaded);
    }

    // 4. Optimizer state
    optimizer.load_state(is);

    // 5. RNG state
    RNGState rng_state;
    is.read(reinterpret_cast<char*>(&rng_state), sizeof(RNGState));
    RNG::set_state(rng_state);
    
    is.close();
}

/**
 * @brief Manages model checkpoints with directory support, auto-discovery,
 * and periodic saving (heartbeats).
 */
class CheckpointManager {
public:
    CheckpointManager(std::string base_dir, std::string prefix = "model", int max_to_keep = 5)
        : base_dir_(base_dir), prefix_(prefix), max_to_keep_(max_to_keep) {
        if (!fs::exists(base_dir_)) {
            fs::create_directories(base_dir_);
        }
        last_save_time_ = std::chrono::steady_clock::now();
    }

    /**
     * @brief Set intervals for automatic saving.
     * @param step_interval Save every N steps (e.g., 100). -1 to disable.
     * @param seconds_interval Save every N seconds (e.g., 1800 for 30min). -1 to disable.
     */
    void set_save_intervals(int step_interval, int seconds_interval = -1) {
        step_interval_ = step_interval;
        seconds_interval_ = seconds_interval;
    }

    /**
     * @brief Smart step function. Saves if either step or time interval is reached.
     * @return true if a checkpoint was saved.
     */
    bool step(int current_step, nn::Module& model, nn::Optimizer& optimizer, float loss) {
        bool should_save = false;

        // Check step-based interval
        if (step_interval_ > 0 && current_step > 0 && current_step % step_interval_ == 0) {
            should_save = true;
        }

        // Check time-based interval (Heartbeat)
        if (seconds_interval_ > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_save_time_).count();
            if (elapsed >= seconds_interval_) {
                std::cout << "[CheckpointManager] Time-based heartbeat triggered after " << elapsed << "s" << std::endl;
                should_save = true;
            }
        }

        if (should_save) {
            save(current_step, model, optimizer, loss);
            return true;
        }
        return false;
    }

    /**
     * @brief Manually save a checkpoint for a specific step.
     */
    void save(int step, nn::Module& model, nn::Optimizer& optimizer, float loss) {
        fs::path p = fs::path(base_dir_) / (prefix_ + "_step_" + std::to_string(step) + ".ckpt");
        std::string path_str = p.string();
        
        save_checkpoint(path_str, model, optimizer, step, loss);
        std::cout << "[CheckpointManager] Saved: " << path_str << std::endl;
        
        last_save_time_ = std::chrono::steady_clock::now();
        cleanup_old_checkpoints();
    }

    /**
     * @brief Automatically find and load the latest checkpoint in base_dir.
     * @return true if a checkpoint was found and loaded.
     */
    bool load_latest(nn::Module& model, nn::Optimizer& optimizer, int& step, float& loss) {
        std::string latest_path = find_latest_checkpoint_path();
        if (latest_path.empty()) {
            return false;
        }

        std::cout << "[CheckpointManager] Auto-loading latest: " << latest_path << std::endl;
        load_checkpoint(latest_path, model, optimizer, step, loss);
        return true;
    }

private:
    std::string find_latest_checkpoint_path() {
        std::vector<std::pair<int, std::string>> checkpoints = list_checkpoints();
        if (checkpoints.empty()) return "";

        // Sort by step number ascending
        std::sort(checkpoints.begin(), checkpoints.end());
        return checkpoints.back().second;
    }

    void cleanup_old_checkpoints() {
        if (max_to_keep_ <= 0) return;

        std::vector<std::pair<int, std::string>> checkpoints = list_checkpoints();
        if (checkpoints.size() <= (size_t)max_to_keep_) return;

        // Sort by step number ascending (oldest first)
        std::sort(checkpoints.begin(), checkpoints.end());

        int to_delete = checkpoints.size() - max_to_keep_;
        for (int i = 0; i < to_delete; ++i) {
            fs::remove(checkpoints[i].second);
            std::cout << "[CheckpointManager] Deleted old checkpoint: " << checkpoints[i].second << std::endl;
        }
    }

    std::vector<std::pair<int, std::string>> list_checkpoints() {
        std::vector<std::pair<int, std::string>> results;
        std::regex re(prefix_ + "_step_(\\d+)\\.ckpt");

        if (!fs::exists(base_dir_)) return {};

        for (const auto& entry : fs::directory_iterator(base_dir_)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, re)) {
                int step = std::stoi(match[1].str());
                results.push_back({step, entry.path().string()});
            }
        }
        return results;
    }

    std::string base_dir_;
    std::string prefix_;
    int max_to_keep_;
    int step_interval_ = -1;
    int seconds_interval_ = -1;
    std::chrono::steady_clock::time_point last_save_time_;
};

} // namespace OwnTensor