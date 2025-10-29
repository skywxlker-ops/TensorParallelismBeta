#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

struct Mesh {
    int world_size;
    std::vector<int> devices;

    Mesh(int ws) : world_size(ws) {
        for (int i = 0; i < ws; ++i) devices.push_back(i);
    }

    void describe() const {
        std::ostringstream oss;
        oss << "[Mesh] Devices: ";
        for (int d : devices) oss << d << " ";
        std::cout << oss.str() << std::endl;
    }
};
