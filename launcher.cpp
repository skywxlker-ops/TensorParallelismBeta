#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <nvml.h>
#include "rendezvous.hpp"

std::map<pid_t, int> pid_to_rank;
std::map<int, int> rank_to_pipe; // Stores the read-end of worker pipes
int master_port = 29877;

void kill_the_gang() {
    if (pid_to_rank.empty()) return;
    std::cout << "\n[Sentinel] Failure Detected. Terminating gang..." << std::endl;
    for (auto const& [pid, rank] : pid_to_rank) {
        kill(-pid, SIGKILL); 
        waitpid(pid, nullptr, 0);
        if (rank_to_pipe.count(rank)) {
            close(rank_to_pipe[rank]);
            rank_to_pipe.erase(rank);
        }
    }
    pid_to_rank.clear();
}

void signal_handler(int sig) { kill_the_gang(); _exit(sig); }

// Helper to drain logs from workers and prefix them
void drain_worker_logs() {
    char buffer[4096];
    for (auto const& [rank, fd] : rank_to_pipe) {
        ssize_t n = read(fd, buffer, sizeof(buffer) - 1);
        if (n > 0) {
            buffer[n] = '\0';
            std::string s(buffer);
            size_t pos = 0;
            // Prefix every new line found in the buffer
            while ((pos = s.find('\n')) != std::string::npos) {
                std::cout << "[Rank " << rank << "] " << s.substr(0, pos + 1);
                s.erase(0, pos + 1);
            }
            if (!s.empty()) std::cout << "[Rank " << rank << "] " << s << std::flush;
        }
    }
}
void launch_worker(int rank, int world_size, const std::vector<std::string>& cmd, int retry, int resolved_port) {
    int pipe_fds[2];
    if (pipe(pipe_fds) == -1) throw std::runtime_error("Pipe failed");

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0); 
        close(pipe_fds[0]);
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);

        // --- ISOLATION: Map Rank to Physical GPU ---
        setenv("CUDA_VISIBLE_DEVICES", std::to_string(rank).c_str(), 1);
        setenv("RANK", std::to_string(rank).c_str(), 1);
        setenv("WORLD_SIZE", std::to_string(world_size).c_str(), 1);
        
        // Pass the REAL port discovered by the Sentinel
        setenv("MASTER_PORT", std::to_string(resolved_port).c_str(), 1);
        setenv("SENTINEL_RETRY", std::to_string(retry).c_str(), 1);

        // Optional: Support Multi-node by passing Master IP if needed
        // setenv("MASTER_ADDR", "127.0.0.1", 0); 

        std::vector<char*> args;
        for (const auto& s : cmd) args.push_back(const_cast<char*>(s.c_str()));
        args.push_back(nullptr);
        
        execvp(args[0], args.data());
        perror("execvp failed"); // Only reached if exec fails
        _exit(1);
    }
    close(pipe_fds[1]); 
    fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK);
    rank_to_pipe[rank] = pipe_fds[0];
    pid_to_rank[pid] = rank;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    int world_size = std::stoi(argv[1]);
    std::vector<std::string> worker_cmd;
    bool found_sep = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--") { found_sep = true; continue; }
        if (found_sep) worker_cmd.push_back(argv[i]);
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        Store sentinel_master(master_port, true);
        int current_port = sentinel_master.actual_port;
        std::cout<<"[Sentinel] Store active on port: "<<current_port<<std::endl;
        int retry_count = 0;
        while (retry_count <= 3) {
            std::cout << "[Sentinel] Launching Cluster (Attempt " << retry_count << ")..." << std::endl;
            for (int i = 0; i < world_size; ++i) launch_worker(i, world_size, worker_cmd, retry_count, current_port);
            
            bool cluster_active = true;
            while (cluster_active && !pid_to_rank.empty()) {
                // 1. Drain and prefix logs
                drain_worker_logs();

                // 2. Check for process exit
                int status;
                pid_t done_pid = waitpid(-1, &status, WNOHANG);
                
                // 3. Check for telemetry-guessed failure
                int guessed_fail = sentinel_master.guess_failed_rank();

                if (done_pid > 0 || guessed_fail != -1) {
                    if (done_pid > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                        int r = pid_to_rank[done_pid];
                        std::cout << "[Sentinel] Rank " << r << " finished successfully." << std::endl;
                        close(rank_to_pipe[r]);
                        rank_to_pipe.erase(r);
                        pid_to_rank.erase(done_pid);
                        continue;
                    }

                    // Something failed
                    kill_the_gang();
                    cluster_active = false;
                    retry_count++;

                    if (retry_count <= 3) {
                        std::cout << "[Sentinel] Waiting for GPUs to clear..." << std::endl;
                        nvmlInit();
                        for (int g = 0; g < world_size; ++g) {
                            nvmlDevice_t dev;
                            nvmlDeviceGetHandleByIndex(g, &dev);
                            uint8_t vram = 100;
                            while (vram > 10) {
                                nvmlMemory_t mem;
                                nvmlDeviceGetMemoryInfo(dev, &mem);
                                vram = (uint8_t)((mem.used * 100) / mem.total);
                                if (vram > 10) usleep(1000000);
                            }
                        }
                        nvmlShutdown();
                        std::cout << "[Sentinel] Ready for relaunch." << std::endl;
                    }
                }
                usleep(50000); // 20Hz refresh for smooth logs
            }
            if (pid_to_rank.empty() && cluster_active) break;
        }
    } catch (...) { kill_the_gang(); }
    return 0;
}
