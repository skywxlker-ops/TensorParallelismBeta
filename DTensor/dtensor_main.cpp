// #include <cuda_runtime.h>
// #include <nccl.h>
// #include <mpi.h>
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <sstream>
// #include <stdexcept>
// #include <cmath>
// #include <cstring>
// #include <iomanip>

// // ================================================================
// //                        Work
// // ================================================================
// class Work {
// public:
//     explicit Work(cudaStream_t s) : stream_(s), done_(false), ok_(true) {
//         cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
//     }
//     ~Work() { cudaEventDestroy(event_); }
//     void markCompleted(bool success = true) {
//         ok_ = success; done_ = true; cudaEventRecord(event_, stream_);
//     }
//     bool wait() { if (!done_) return false; cudaEventSynchronize(event_); return ok_; }
// private:
//     cudaStream_t stream_;
//     cudaEvent_t event_;
//     bool done_, ok_;
// };

// // ================================================================
// //                        DeviceMesh
// // ================================================================
// class DeviceMesh {
// public:
//     DeviceMesh(std::vector<int> devices, std::vector<int> shape)
//         : devices_(std::move(devices)), shape_(std::move(shape)) {
//         if (devices_.size() != product(shape_))
//             throw std::runtime_error("Mesh shape/device mismatch");
//     }

//     static int product(const std::vector<int>& d) { int p=1; for (int x:d) p*=x; return p; }

//     std::vector<int> coordsForDevice(int rank) const {
//         std::vector<int> c(shape_.size());
//         int idx=rank;
//         for (int i=shape_.size()-1;i>=0;--i){ c[i]=idx%shape_[i]; idx/=shape_[i]; }
//         return c;
//     }

//     std::string toJSON() const {
//         std::ostringstream ss;
//         ss << "{\n  \"shape\": [";
//         for (size_t i=0;i<shape_.size();++i){ ss<<shape_[i]; if(i+1<shape_.size()) ss<<", "; }
//         ss << "],\n  \"devices\": [";
//         for (size_t i=0;i<devices_.size();++i){ ss<<devices_[i]; if(i+1<devices_.size()) ss<<", "; }
//         ss << "]\n}";
//         return ss.str();
//     }

//     const std::vector<int>& shape() const { return shape_; }
//     int size() const { return devices_.size(); }

// private:
//     std::vector<int> devices_;
//     std::vector<int> shape_;
// };

// // ================================================================
// //                        Layout
// // ================================================================
// enum class PlacementType { Shard, Replicate, Partial };

// class Layout {
// public:
//     Layout(std::vector<PlacementType> p, const DeviceMesh& m)
//         : placements_(std::move(p)), mesh_(m) {}

//     std::string toJSON() const {
//         std::ostringstream s;
//         s << "{\n  \"placements\": [";
//         for(size_t i=0;i<placements_.size();++i){
//             s << "\"" << placementName(placements_[i]) << "\"";
//             if(i+1<placements_.size()) s << ", ";
//         }
//         s << "],\n  \"mesh\": " << mesh_.toJSON() << "\n}";
//         return s.str();
//     }

// private:
//     static std::string placementName(PlacementType p) {
//         switch(p){
//             case PlacementType::Shard: return "Shard";
//             case PlacementType::Replicate: return "Replicate";
//             case PlacementType::Partial: return "Partial";
//         }
//         return "Unknown";
//     }

//     std::vector<PlacementType> placements_;
//     const DeviceMesh& mesh_;
// };

// // ================================================================
// //                        TensorSpec
// // ================================================================
// struct TensorSpec {
//     std::vector<int> global_shape;
//     std::vector<int> local_shape;
//     std::vector<int> offsets;
//     std::string toString() const {
//         std::ostringstream ss;
//         ss << "TensorSpec(global=";
//         for(auto v:global_shape) ss<<v<<",";
//         ss << " local=";
//         for(auto v:local_shape) ss<<v<<",";
//         ss << " offsets=";
//         for(auto v:offsets) ss<<v<<",";
//         ss << ")";
//         return ss.str();
//     }
// };

// // ================================================================
// //                        ProcessGroup
// // ================================================================
// class ProcessGroup {
// public:
//     ProcessGroup(int rank,int world,int device,const ncclUniqueId& id)
//         : rank_(rank), world_(world) {
//         cudaSetDevice(device);
//         cudaStreamCreate(&stream_);
//         ncclCommInitRank(&comm_, world_, id, rank_);
//     }
//     ~ProcessGroup(){ ncclCommDestroy(comm_); cudaStreamDestroy(stream_); }

//     template<typename T>
//     std::shared_ptr<Work> allReduce(T* data,size_t n,ncclDataType_t t){
//         auto w=std::make_shared<Work>(stream_);
//         ncclAllReduce(data,data,n,t,ncclSum,comm_,stream_);
//         w->markCompleted(true); return w;
//     }

//     template<typename T>
//     std::shared_ptr<Work> broadcast(T* data,size_t n,ncclDataType_t t,int root){
//         auto w=std::make_shared<Work>(stream_);
//         ncclBroadcast(data,data,n,t,root,comm_,stream_);
//         w->markCompleted(true); return w;
//     }

//     template<typename T>
//     std::shared_ptr<Work> allGather(T* sendbuf,T* recvbuf,size_t n,ncclDataType_t t){
//         auto w=std::make_shared<Work>(stream_);
//         ncclAllGather(sendbuf,recvbuf,n,t,comm_,stream_);
//         w->markCompleted(true); return w;
//     }

//     template<typename T>
//     std::shared_ptr<Work> reduceScatter(T* sendbuf,T* recvbuf,size_t n,ncclDataType_t t){
//         auto w=std::make_shared<Work>(stream_);
//         ncclReduceScatter(sendbuf,recvbuf,n,t,ncclSum,comm_,stream_);
//         w->markCompleted(true); return w;
//     }

// private:
//     int rank_, world_;
//     ncclComm_t comm_;
//     cudaStream_t stream_;
// };

// // ================================================================
// //                        DTensor
// // ================================================================
// class DTensor {
// public:
//     DTensor(int world,int sz,int rank):world_(world),sz_(sz),rank_(rank){
//         h_.resize(sz_);
//         for(int j=0;j<sz_;++j) h_[j]=float(rank_*sz_+j);
//         cudaMalloc(&d_,sz_*sizeof(float));
//         cudaMemcpy(d_,h_.data(),sz_*sizeof(float),cudaMemcpyHostToDevice);
//     }
//     ~DTensor(){ if(d_) cudaFree(d_); }
//     float* d() { return d_; }
//     void toHost(){ cudaMemcpy(h_.data(),d_,sz_*sizeof(float),cudaMemcpyDeviceToHost); }
//     std::string str() const {
//         std::stringstream s; s<<"[Rank "<<rank_<<"] ";
//         for(float v:h_) s<<v<<" "; return s.str();
//     }
// private:
//     int world_,sz_,rank_;
//     std::vector<float> h_;
//     float* d_=nullptr;
// };

// // ================================================================
// //                        Helpers
// // ================================================================
// void barrierHeader(int rank,const std::string& h){
//     MPI_Barrier(MPI_COMM_WORLD);
//     if(rank==0){ std::cout<<"\n"<<std::string(60,'=')<<"\n"<<h<<"\n"; }
//     MPI_Barrier(MPI_COMM_WORLD);
// }
// void gatherPrint(int rank,int world,const std::string& d){
//     std::vector<int> sz(world);
//     int my=d.size(); MPI_Gather(&my,1,MPI_INT,sz.data(),1,MPI_INT,0,MPI_COMM_WORLD);
//     if(rank==0){
//         std::vector<std::string> all(world);
//         all[0]=d;
//         for(int r=1;r<world;r++){ all[r].resize(sz[r]); MPI_Recv(&all[r][0],sz[r],MPI_CHAR,r,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
//         for(auto& s:all) std::cout<<s<<"\n";
//     } else MPI_Send(d.c_str(),my,MPI_CHAR,0,0,MPI_COMM_WORLD);
//     MPI_Barrier(MPI_COMM_WORLD);
// }

// // ================================================================
// //                        Planner
// // ================================================================
// struct Planner {
//     static DeviceMesh inferMesh() {
//         int count=0; cudaGetDeviceCount(&count);
//         if(count<=0) throw std::runtime_error("No GPUs found");
//         std::vector<int> dev(count); for(int i=0;i<count;++i) dev[i]=i;
//         std::vector<int> shape;
//         if(count==8) shape={2,4};
//         else if(count==4) shape={2,2};
//         else shape={count};
//         return DeviceMesh(dev,shape);
//     }
//     static Layout inferLayout(const DeviceMesh& mesh){
//         std::vector<PlacementType> p;
//         for(size_t i=0;i<mesh.shape().size();++i)
//             p.push_back(i==0?PlacementType::Shard:PlacementType::Replicate);
//         return Layout(p,mesh);
//     }
// };

// // ================================================================
// //                        Worker
// // ================================================================
// void worker(int rank,int world,const ncclUniqueId& id,bool printMesh,bool printLayout){
//     ProcessGroup pg(rank,world,rank,id);
//     auto mesh=Planner::inferMesh();
//     auto layout=Planner::inferLayout(mesh);

//     if(printMesh && rank==0) std::cout<<mesh.toJSON()<<"\n";
//     if(printLayout && rank==0) std::cout<<layout.toJSON()<<"\n";

//     DTensor t(world,8,rank);
//     barrierHeader(rank,"BEFORE ALLREDUCE");
//     gatherPrint(rank,world,t.str());

//     pg.allReduce(t.d(),8,ncclFloat32)->wait();
//     t.toHost();
//     barrierHeader(rank,"AFTER ALLREDUCE");
//     gatherPrint(rank,world,t.str());
// }

// // ================================================================
// //                        Main
// // ================================================================
// int main(int argc,char** argv){
//     MPI_Init(&argc,&argv);
//     int rank,world; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&world);
//     ncclUniqueId id;
//     if(rank==0) ncclGetUniqueId(&id);
//     MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);

//     bool printMesh=false, printLayout=false;
//     for(int i=1;i<argc;++i){
//         if(strcmp(argv[i],"--print-mesh")==0) printMesh=true;
//         if(strcmp(argv[i],"--print-layout")==0) printLayout=true;
//     }

//     if(rank==0) std::cout<<"Starting DTensor with automatic mesh/layout detection\n";
//     worker(rank,world,id,printMesh,printLayout);
//     MPI_Finalize();
//     return 0;
// }



#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cmath>

// ================================================================
//                        Work
// ================================================================
class Work {
public:
    explicit Work(cudaStream_t s) : stream_(s), done_(false), ok_(true) {
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    }
    ~Work() { cudaEventDestroy(event_); }
    void markCompleted(bool success = true) {
        ok_ = success; done_ = true; cudaEventRecord(event_, stream_);
    }
    bool wait() { if (!done_) return false; cudaEventSynchronize(event_); return ok_; }
private:
    cudaStream_t stream_;
    cudaEvent_t event_;
    bool done_, ok_;
};

// ================================================================
//                        DeviceMesh
// ================================================================
class DeviceMesh {
public:
    DeviceMesh(std::vector<int> devices, std::vector<int> shape)
        : devices_(std::move(devices)), shape_(std::move(shape)) {
        if (devices_.size() != product(shape_))
            throw std::runtime_error("Mesh shape/device mismatch");
    }

    static int product(const std::vector<int>& d) { int p=1; for (int x:d) p*=x; return p; }

    std::vector<int> coordsForDevice(int rank) const {
        std::vector<int> c(shape_.size());
        int idx=rank;
        for (int i=shape_.size()-1;i>=0;--i){ c[i]=idx%shape_[i]; idx/=shape_[i]; }
        return c;
    }

    std::string describeJSON() const {
        std::stringstream s;
        s<<"{\"shape\": [";
        for (size_t i=0;i<shape_.size();++i){ s<<shape_[i]; if(i+1<shape_.size()) s<<","; }
        s<<"], \"devices\": [";
        for (size_t i=0;i<devices_.size();++i){ s<<devices_[i]; if(i+1<devices_.size()) s<<","; }
        s<<"]}";
        return s.str();
    }

    const std::vector<int>& shape() const { return shape_; }
    int size() const { return devices_.size(); }

private:
    std::vector<int> devices_;
    std::vector<int> shape_;
};

// ================================================================
//                        Layout
// ================================================================
enum class PlacementType { Shard, Replicate, Partial };

class Layout {
public:
    Layout(std::vector<PlacementType> p, const DeviceMesh& m)
        : placements_(std::move(p)), mesh_(m) {}

    std::string describeJSON() const {
        std::stringstream s;
        s<<"{\"placements\": [";
        for(size_t i=0;i<placements_.size();++i){
            switch(placements_[i]){
                case PlacementType::Shard: s<<"\"Shard\""; break;
                case PlacementType::Replicate: s<<"\"Replicate\""; break;
                case PlacementType::Partial: s<<"\"Partial\""; break;
            }
            if(i+1<placements_.size()) s<<",";
        }
        s<<"], \"mesh\": "<<mesh_.describeJSON()<<"}";
        return s.str();
    }
private:
    std::vector<PlacementType> placements_;
    const DeviceMesh& mesh_;
};

// ================================================================
//                        ProcessGroup
// ================================================================
class ProcessGroup {
public:
    ProcessGroup(int rank,int world,int device,const ncclUniqueId& id)
        : rank_(rank), world_(world) {
        cudaSetDevice(device);
        cudaStreamCreate(&stream_);
        ncclCommInitRank(&comm_, world_, id, rank_);
    }
    ~ProcessGroup(){ ncclCommDestroy(comm_); cudaStreamDestroy(stream_); }

    template<typename T>
    std::shared_ptr<Work> allReduce(T* data,size_t n,ncclDataType_t t){
        auto w=std::make_shared<Work>(stream_);
        ncclAllReduce(data,data,n,t,ncclSum,comm_,stream_);
        w->markCompleted(true); return w;
    }

    template<typename T>
    std::shared_ptr<Work> allReduceUneven(T* data, size_t count, size_t max_count, ncclDataType_t dtype) {
        T* d_padded = data;
        if (count < max_count) {
            cudaMalloc(&d_padded, max_count * sizeof(T));
            cudaMemcpy(d_padded, data, count * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemset(d_padded + count, 0, (max_count - count) * sizeof(T));
        }

        auto w = std::make_shared<Work>(stream_);
        ncclAllReduce(d_padded, d_padded, max_count, dtype, ncclSum, comm_, stream_);
        w->markCompleted(true);
        w->wait();

        if (count < max_count) {
            cudaMemcpy(data, d_padded, count * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaFree(d_padded);
        }

        return w;
    }

private:
    int rank_, world_;
    ncclComm_t comm_;
    cudaStream_t stream_;
};

// ================================================================
//                        DTensor
// ================================================================
class DTensor {
public:
    DTensor(int world, int global_size, int rank, bool pad_mode)
        : world_size_(world), global_size_(global_size), rank_(rank), pad_mode_(pad_mode) {

        base_ = global_size_ / world_size_;
        extra_ = global_size_ % world_size_;

        if (pad_mode_) {
            // Pad total size so all shards equal
            local_size_ = base_ + (extra_ ? 1 : 0);
            padded_global_size_ = local_size_ * world_size_;
            offset_ = rank_ * local_size_;
        } else {
            // Uneven shard split
            local_size_ = base_ + (rank_ < extra_ ? 1 : 0);
            offset_ = rank_ * base_ + std::min(rank_, extra_);
            padded_global_size_ = global_size_;
        }

        h_.resize(local_size_);
        for (int i = 0; i < local_size_; ++i)
            h_[i] = float(offset_ + i);

        cudaMalloc(&d_, local_size_ * sizeof(float));
        cudaMemcpy(d_, h_.data(), local_size_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~DTensor(){ if(d_) cudaFree(d_); }

    float* data() { return d_; }
    int localSize() const { return local_size_; }
    int offset() const { return offset_; }
    bool isPadded() const { return pad_mode_; }

    void toHost(){ cudaMemcpy(h_.data(), d_, local_size_ * sizeof(float), cudaMemcpyDeviceToHost); }

    std::string str() const {
        std::stringstream s;
        s<<"[Rank "<<rank_<<" | Offset "<<offset_<<" | Size "<<local_size_;
        if (pad_mode_) s<<" | Mode=Padded";
        else s<<" | Mode=Uneven";
        s<<"] ";
        for(float v:h_) s<<v<<" ";
        return s.str();
    }

private:
    int world_size_, global_size_, rank_;
    int base_, extra_;
    int local_size_, offset_;
    int padded_global_size_;
    bool pad_mode_;
    std::vector<float> h_;
    float* d_=nullptr;
};

// ================================================================
//                        Helpers
// ================================================================
void barrierHeader(int rank,const std::string& h){
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){ std::cout<<"\n"<<std::string(60,'=')<<"\n"<<h<<"\n"; }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gatherPrint(int rank,int world,const std::string& d){
    std::vector<int> sz(world);
    int my=d.size(); MPI_Gather(&my,1,MPI_INT,sz.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank==0){
        std::vector<std::string> all(world);
        all[0]=d;
        for(int r=1;r<world;r++){ all[r].resize(sz[r]); MPI_Recv(&all[r][0],sz[r],MPI_CHAR,r,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); }
        for(auto& s:all) std::cout<<s<<"\n";
    } else MPI_Send(d.c_str(),my,MPI_CHAR,0,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

// ================================================================
//                        Adaptive Planner
// ================================================================
struct Planner {
    static DeviceMesh inferMesh() {
        int count=0; cudaGetDeviceCount(&count);
        if(count<=0) throw std::runtime_error("No GPUs found");
        std::vector<int> dev(count); for(int i=0;i<count;++i) dev[i]=i;
        std::vector<int> shape;
        if(count==8) shape={2,4};
        else if(count==4) shape={2,2};
        else shape={count};
        return DeviceMesh(dev,shape);
    }

    static Layout inferLayout(const DeviceMesh& mesh){
        std::vector<PlacementType> p;
        for(size_t i=0;i<mesh.shape().size();++i)
            p.push_back(i==0?PlacementType::Shard:PlacementType::Replicate);
        return Layout(p,mesh);
    }

    // Core: automatically decide how to shard
    static bool shouldPadShard(int global_size, int world_size){
        int remainder = global_size % world_size;
        float ratio = float(remainder) / world_size;
        // Pad if imbalance exceeds ~20%
        return ratio > 0.2f;
    }
};

// ================================================================
//                        Worker
// ================================================================
void worker(int rank,int world,const ncclUniqueId& id){
    ProcessGroup pg(rank,world,rank,id);

    auto mesh=Planner::inferMesh();
    auto layout=Planner::inferLayout(mesh);

    if(rank==0)
        std::cout<<"{\n  \"mesh\": "<<mesh.describeJSON()<<",\n  \"layout\": "<<layout.describeJSON()<<"\n}\n";

    int global_size = 25; // intentionally uneven
    bool pad_mode = Planner::shouldPadShard(global_size, world);

    DTensor t(world, global_size, rank, pad_mode);

    barrierHeader(rank,"BEFORE ALLREDUCE");
    gatherPrint(rank,world,t.str());

    // Determine max shard size
    size_t local_size = t.localSize(), max_size;
    MPI_Allreduce(&local_size, &max_size, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (pad_mode)
        pg.allReduce(t.data(), t.localSize(), ncclFloat32)->wait();
    else
        pg.allReduceUneven(t.data(), t.localSize(), max_size, ncclFloat32)->wait();

    t.toHost();

    barrierHeader(rank,"AFTER ALLREDUCE");
    gatherPrint(rank,world,t.str());
}

// ================================================================
//                        Main
// ================================================================
int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int rank,world; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&world);
    ncclUniqueId id;
    if(rank==0) ncclGetUniqueId(&id);
    MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);

    if(rank==0) std::cout<<"Starting DTensor with adaptive sharding\n";
    worker(rank,world,id);
    MPI_Finalize();
    return 0;
}
