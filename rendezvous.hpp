#ifndef RENDEZVOUS_HPP
#define RENDEZVOUS_HPP

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/poll.h>
#include <nvml.h> 
#include <chrono>

enum class MsgType : uint8_t { JOIN = 1, SET = 2, GET = 3, COMMIT = 4, EVENT_RECONFIG = 5, GET_WORLD_SIZE = 6 };

#pragma pack(push, 1)
struct Packet {
    MsgType type;
    uint8_t rank;        
    uint16_t epoch;      
    uint8_t gpu_util;     
    uint8_t vram_util;    
    uint16_t gpu_temp;    
    uint32_t key_len;
    uint32_t val_len;
};
#pragma pack(pop)

class Store {
private:
    int _fd = -1;
    int _my_rank = -1;
    bool _is_master;
    std::atomic<bool> _running{true};
    std::map<std::string, std::vector<uint8_t>> _kv_store;
    std::mutex _kv_mtx;
    std::condition_variable _kv_cv;
    std::thread _bg_thread;

    struct HealthNode { 
        int fd; 
        std::chrono::steady_clock::time_point last_seen;
        uint8_t last_vram;
    };
    std::map<int, HealthNode> _ranks; 
    std::mutex _mgmt_mtx; 

    nvmlDevice_t _nv_handle;
    bool _nv_active = false;

    void _sync_telemetry(Packet& p) {
        if(!_nv_active) return;
        nvmlUtilization_t util;
        nvmlMemory_t mem;
        unsigned int temp;
        if(nvmlDeviceGetUtilizationRates(_nv_handle, &util) == NVML_SUCCESS) p.gpu_util = (uint8_t)util.gpu;
        if(nvmlDeviceGetMemoryInfo(_nv_handle, &mem) == NVML_SUCCESS) p.vram_util = (uint8_t)((mem.used * 100) / mem.total);
        if(nvmlDeviceGetTemperature(_nv_handle, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) p.gpu_temp = (uint16_t)temp;
    }

    void _send_all(int fd, const void* data, size_t len) {
        size_t total = 0;
        while(total < len) {
            ssize_t n = send(fd, (const char*)data + total, len - total, MSG_NOSIGNAL);
            if (n <= 0) return;
            total += n;
        }
    }

    void _recv_all(int fd, void* data, size_t len) {
        size_t total = 0;
        while(total < len) {
            ssize_t n = recv(fd, (char*)data + total, len - total, 0);
            if (n <= 0) throw std::runtime_error("Recv failed");
            total += n;
        }
    }

    void _process_packet(int fd, Packet pkt) {
        {
            std::lock_guard<std::mutex> lock(_mgmt_mtx);
            _ranks[pkt.rank] = {fd, std::chrono::steady_clock::now(), pkt.vram_util};
        }
        uint32_t k_len = ntohl(pkt.key_len);
        uint32_t v_len = ntohl(pkt.val_len);

        if (pkt.type == MsgType::SET) {
            std::vector<char> k_buf(k_len); _recv_all(fd, k_buf.data(), k_len);
            std::vector<uint8_t> v_buf(v_len); _recv_all(fd, v_buf.data(), v_len);
            std::lock_guard<std::mutex> lock(_kv_mtx);
            _kv_store[std::string(k_buf.begin(), k_buf.end())] = v_buf;
            _kv_cv.notify_all();
        } 
        else if (pkt.type == MsgType::GET) {
            std::vector<char> k_buf(k_len); _recv_all(fd, k_buf.data(), k_len);
            std::string key(k_buf.begin(), k_buf.end());
            std::unique_lock<std::mutex> lock(_kv_mtx);
            _kv_cv.wait(lock, [&]{ return _kv_store.count(key) || !_running; });
            auto& val = _kv_store[key];
            uint32_t net_len = htonl((uint32_t)val.size());
            _send_all(fd, &net_len, sizeof(net_len));
            _send_all(fd, val.data(), val.size());
        }
    }

    void _master_loop() {
        std::vector<pollfd> fds;
        fds.push_back({_fd, POLLIN, 0});

        while (_running) {
            int ret = poll(fds.data(), (nfds_t)fds.size(), 500);
            if (ret < 0) break;
            if (fds[0].revents & POLLIN) {
                int client = accept(_fd, nullptr, nullptr);
                if (client >= 0) fds.push_back({client, POLLIN, 0});
            }
            for (size_t i = 1; i < fds.size(); ++i) {
                if (fds[i].revents & POLLIN) {
                    Packet pkt;
                    if (recv(fds[i].fd, &pkt, sizeof(pkt), 0) <= 0) {
                        close(fds[i].fd);
                        fds.erase(fds.begin() + i--);
                        continue;
                    }
                    _process_packet(fds[i].fd, pkt);
                }
            }
        }
    }

public:
    int actual_port = -1;

    Store(int port, bool is_master, std::string addr = "127.0.0.1", int rank = -1) 
        : _my_rank(rank), _is_master(is_master) {
        if (_is_master) {
            _fd = socket(AF_INET, SOCK_STREAM, 0);
            
            int opt = 1; 
            setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            struct linger sl = {1, 0}; // l_onoff = 1, l_linger = 0 (Hard Reset)
            setsockopt(_fd, SOL_SOCKET, SO_LINGER, &sl, sizeof(sl));

            sockaddr_in srv{}; 
            srv.sin_family = AF_INET; 
            srv.sin_port = htons(port); 
            srv.sin_addr.s_addr = INADDR_ANY;

            if (bind(_fd, (struct sockaddr*)&srv, sizeof(srv)) < 0) 
                throw std::runtime_error("Bind failed: " + std::string(strerror(errno)));
            
            // 2. Port Discovery: If port was 0, find out what the OS gave us
            socklen_t srv_len = sizeof(srv);
            getsockname(_fd, (struct sockaddr*)&srv, &srv_len);
            actual_port = ntohs(srv.sin_port);

            listen(_fd, 128);
            _bg_thread = std::thread(&Store::_master_loop, this);
        } else {
            if(nvmlInit() == NVML_SUCCESS) {
                if(nvmlDeviceGetHandleByIndex(_my_rank, &_nv_handle) == NVML_SUCCESS) _nv_active = true;
            }
            _fd = socket(AF_INET, SOCK_STREAM, 0);
            
            struct linger sl = {1, 0};
            setsockopt(_fd, SOL_SOCKET, SO_LINGER, &sl, sizeof(sl));

            sockaddr_in srv{}; 
            srv.sin_family = AF_INET; 
            srv.sin_port = htons(port);
            inet_pton(AF_INET, addr.c_str(), &srv.sin_addr);

            // Robust connect loop
            int retries = 0;
            while(connect(_fd, (struct sockaddr*)&srv, sizeof(srv)) < 0) {
                if (++retries > 500) throw std::runtime_error("Worker failed to connect to Master");
                usleep(10000);
            }
            actual_port = port;

            Packet p{}; p.type = MsgType::JOIN; p.rank = (uint8_t)rank;
            _sync_telemetry(p);
            _send_all(_fd, &p, sizeof(p));
        }
    }


    void set(const std::string& key, const std::vector<uint8_t>& val) {
        Packet p{}; p.type = MsgType::SET; p.rank = (uint8_t)_my_rank;
        _sync_telemetry(p);
        p.key_len = htonl(key.size()); p.val_len = htonl(val.size());
        _send_all(_fd, &p, sizeof(p));
        _send_all(_fd, key.data(), key.size());
        _send_all(_fd, val.data(), val.size());
    }

    std::vector<uint8_t> get(const std::string& key) {
        Packet p{}; p.type = MsgType::GET; p.rank = (uint8_t)_my_rank;
        _sync_telemetry(p);
        p.key_len = htonl(key.size());
        _send_all(_fd, &p, sizeof(p));
        _send_all(_fd, key.data(), key.size());
        uint32_t len; _recv_all(_fd, &len, sizeof(len));
        std::vector<uint8_t> val(ntohl(len));
        _recv_all(_fd, val.data(), val.size());
        return val;
    }

    int guess_failed_rank() {
        std::lock_guard<std::mutex> lock(_mgmt_mtx);
        auto now = std::chrono::steady_clock::now();
        for (auto const& [rank, node] : _ranks) {
            auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - node.last_seen).count();
            if (diff > 3 && node.last_vram > 95) return rank;
            if (diff > 15) return rank; 
        }
        return -1;
    }

    void shutdown() { _running = false; if (_bg_thread.joinable()) _bg_thread.join(); if (_fd >= 0) close(_fd); }
    ~Store() { if (_nv_active) nvmlShutdown(); shutdown(); }
};
#endif
