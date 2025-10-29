#pragma once
#include <vector>
#include <string>
#include <sstream>

struct Layout {
    int rank;
    int world_size;
    std::vector<int> shard_sizes;

    Layout(int r, int ws, const std::vector<int>& ss)
        : rank(r), world_size(ws), shard_sizes(ss) {}

    std::string describe() const {
        std::ostringstream oss;
        oss << "[Layout] Rank " << rank << "/" << world_size
            << " owns shard of size " << shard_sizes[rank];
        return oss.str();
    }
};
