#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

struct Layout {
    int global_size;
    int world_size;
    std::vector<int> shard_sizes;
    int shard_dim;
};

struct Planner {
    static std::vector<int> shardSizes(int global_size, int world_size) {
        int base = global_size / world_size;
        int rem = global_size % world_size;
        std::vector<int> sizes(world_size, base);
        for (int i = 0; i < rem; ++i) sizes[i]++;
        return sizes;
    }

    static bool shouldPadShard(int global_size, int world_size) {
        return (global_size % world_size) != 0;
    }

    static Layout inferLayout(int global_size, int world_size, int shard_dim = 0) {
        Layout l;
        l.global_size = global_size;
        l.world_size = world_size;
        l.shard_sizes = shardSizes(global_size, world_size);
        l.shard_dim = shard_dim;
        return l;
    }

    static std::string describePlan(const Layout& layout) {
        std::ostringstream oss;
        oss << "[Planner] Global size = " << layout.global_size << "\n";
        oss << "[Planner] World size = " << layout.world_size << "\n";
        oss << "[Planner] Shard sizes = ";
        for (auto s : layout.shard_sizes) oss << s << " ";
        oss << "\n[Planner] Shard dim = " << layout.shard_dim << "\n";
        oss << "[Planner] Padding required = "
            << (shouldPadShard(layout.global_size, layout.world_size) ? "true" : "false") << "\n";
        return oss.str();
    }

    static void printLayoutJSON(int rank, int world_size, int global_size) {
        auto layout = inferLayout(global_size, world_size);
        std::cout << "{\n"
                  << "  \"rank\": " << rank << ",\n"
                  << "  \"world_size\": " << world_size << ",\n"
                  << "  \"device\": " << rank << ",\n"
                  << "  \"mesh\": [";
        for (int i = 0; i < world_size; i++) {
            std::cout << i;
            if (i < world_size - 1) std::cout << ", ";
        }
        std::cout << "],\n"
                  << "  \"placement\": \"rank_" << rank << "_on_device_" << rank << "\",\n"
                  << "  \"layout\": {\n"
                  << "    \"global_size\": " << layout.global_size << ",\n"
                  << "    \"local_shard_size\": " << layout.shard_sizes[rank] << ",\n"
                  << "    \"shard_dim\": " << layout.shard_dim << "\n"
                  << "  }\n"
                  << "}" << std::endl;
    }
};
