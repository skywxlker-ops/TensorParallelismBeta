#include "ProcessGroupNCCL.h"
#include <iostream>
#include <vector>
#include <memory.h>
#include <cstdint>
#include <future>
#include <chrono>

#define TIME_INTERVAL 10

#define RESULT_CHECK(cmd)                                                  \
    do{                                                                    \
        result_t r = (cmd);                                                \
        if(r != pgSuccess){                                                \
            throw std::runtime_error(std::string("Process Group Error"));  \
        }                                                                  \
    } while(0)                                                             \




ProcessGroupNCCL::ProcessGroupNCCL(int world_size, int rank, ncclUniqueId& id, std::shared_ptr<Work> work_obj, cudaStream_t& stream )
:world_size_(world_size), id_(id), rank_(rank), work_obj_(work_obj), communication_stream_(stream)
{
    //no_of_gpus per node for initialization.

    const char* env = std::getenv("NO_GPUS_PER_NODE");
    if (env) {
        int parsed = std::atoi(env);
        if (parsed > 0) gpus_per_node_ = parsed;
    }


    local_rank_ = rank % gpus_per_node_;
    // CUDACHECK(cudaSetDevice(local_rank_)); // Removed: Trust main's cudaSetDevice

    NCCLCHECK(ncclCommInitRank(&comm_, world_size_, id_, rank_));

    work_obj_ -> setCommunicator(comm_);
    if(comm_ == (ncclComm_t)NULL){
        throw std::runtime_error(std::string("NCCL COMMUNICATION NOT INITIALIZED"));
    }
    
}

ProcessGroupNCCL::~ProcessGroupNCCL(){
    NCCLCHECK(ncclCommDestroy(comm_));
    if(owns_stream_){
        CUDACHECK(cudaStreamDestroy(communication_stream_));
    }
}






// void ProcessGroupNCCL::start_time(){
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start,stream_);
// }

// void ProcessGroupNCCL::end_time(float& ms){
//     cudaEventRecord(stop, stream_);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&ms, start_, stop_);
//     cudaEventDestroy(start_);
//     cudaEventDestroy(stop_);
// }


void ProcessGroupNCCL::start_time() {
    // Create events fresh for each measurement to avoid resource handle errors.
    CUDACHECK(cudaEventCreate(&start_));
    CUDACHECK(cudaEventCreate(&stop_));
    CUDACHECK(cudaEventRecord(start_, communication_stream_));
}

void ProcessGroupNCCL::end_time(float& ms) {
    CUDACHECK(cudaEventRecord(stop_, communication_stream_));
    CUDACHECK(cudaEventSynchronize(stop_));
    CUDACHECK(cudaEventElapsedTime(&ms, start_, stop_));

    // Destroy events after use.
    CUDACHECK(cudaEventDestroy(start_));
    CUDACHECK(cudaEventDestroy(stop_));
    start_ = nullptr;
    stop_ = nullptr;
}


//init commuication
std::shared_ptr<ProcessGroupNCCL> init_process_group(int world_size, int rank, cudaStream_t stream){
    cudaStream_t communication_stream;
    bool stream_created = false;
    if(stream == 0){
        CUDACHECK(cudaStreamCreate(&communication_stream));
        stream_created = true;
    }else{
        communication_stream = stream;
    }
    ncclUniqueId id;
    if(rank == 0){
        NCCLCHECK(ncclGetUniqueId(&id));
        //to communicate the id to all the processes.
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::shared_ptr<Work> work_obj = std::make_shared<Work>(communication_stream, nullptr);
    auto pg = std::make_shared<ProcessGroupNCCL>(world_size, rank, id, work_obj, communication_stream);
    pg->set_owns_stream(stream_created);
    return pg;
}

bool ProcessGroupNCCL::blockStreamEvent(){
    //stream synchronize is for the entire stream. but event synchronize is only upto a point
    work_obj_->wait();
    if(work_obj_->is_success() == false){
        return false;
    }
    return true;
}

bool ProcessGroupNCCL::blockStream(){
    CUDACHECK(cudaStreamSynchronize(communication_stream_));
    return true;
}


// result_t ToNcclError(bool status, std::shared_ptr<Work> work_obj){
//     if(status == true){
//         return pgSuccess;
//     }else{
//         if(work)
//     }
// }


//collectives


//all reduce
result_t ProcessGroupNCCL::all_reduce(const void* sendbuff, void* recvbuff,size_t count, OwnTensor::Dtype dtype, op_t operation, bool sync){
    auto work_obj = all_reduce_async(sendbuff, recvbuff, count, dtype, operation, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

// 
result_t ProcessGroupNCCL::reduce_scatter(const void* sendbuff, void* recvbuff, size_t recv_count, OwnTensor::Dtype dtype, op_t operation, bool sync){
    auto work_obj = reduce_scatter_async(sendbuff, recvbuff, recv_count, dtype, operation, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

result_t ProcessGroupNCCL::all_gather(const void* sendbuff, void* recvbuff, size_t sendcount, OwnTensor::Dtype dtype, bool sync){
    auto work_obj = all_gather_async(sendbuff, recvbuff, sendcount, dtype, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

result_t ProcessGroupNCCL::gather(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync){
    auto work_obj = gather_async(sendbuff, recvbuff, count, dtype, root, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

result_t ProcessGroupNCCL::reduce(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, op_t op, int root, bool sync){
    auto work_obj = reduce_async(sendbuff, recvbuff, count, dtype, op, root, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

result_t ProcessGroupNCCL::scatter(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync){
    auto work_obj = scatter_async(sendbuff, recvbuff, count, dtype, root, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

result_t ProcessGroupNCCL::broadcast(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync){
    auto work_obj = broadcast_async(sendbuff, recvbuff, count, dtype, root, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

result_t ProcessGroupNCCL::alltoall(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, bool sync){
    auto work_obj = alltoall_async(sendbuff, recvbuff, count, dtype, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}


result_t ProcessGroupNCCL::sendrecv(const void* sendbuff, void* recvbuff, int send_rank, int recv_rank, size_t count, OwnTensor::Dtype dtype, bool sync){
    auto work_obj = sendrecv_async(sendbuff, recvbuff, send_rank, recv_rank, count, dtype, sync);

    if(work_obj -> is_success()) return pgSuccess;
    else{
        if(work_obj -> get_lastCudaError() != cudaSuccess ) return pgCudaError; 
        else if( work_obj -> get_ncclStatus() != ncclSuccess ) return pgNcclError;
        else if( time_limit_exceeded.load() ) return pgTimeout;
    }
    return pgInternalError;
}

//all reduce
std::shared_ptr<Work> ProcessGroupNCCL::all_reduce_async(const void* sendbuff, void* recvbuff,size_t count, OwnTensor::Dtype dtype, op_t operation, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclAllReduce(sendbuff, recvbuff, count, ncclTypeConversion(dtype), ncclOperationConversion(operation), comm_, communication_stream_);
        },
        sync_
    );
}


//reduce
std::shared_ptr<Work> ProcessGroupNCCL::reduce_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, op_t op, int root, bool sync_){
    // while(!work_obj_->is_completed());

    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclReduce(sendbuff, recvbuff, count, ncclTypeConversion(dtype), ncclOperationConversion(op), root,  comm_, communication_stream_);
        },
        sync_
    );
}

//reduce scatter
 
std::shared_ptr<Work> ProcessGroupNCCL::reduce_scatter_async(const void* sendbuff, void* recvbuff, size_t recv_count, OwnTensor::Dtype dtype, op_t operation, bool sync_){
    // while(!work_obj_->is_completed());

    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclReduceScatter(sendbuff, recvbuff, recv_count, ncclTypeConversion(dtype), ncclOperationConversion(operation), comm_, communication_stream_);
        },
        sync_
    );
}


//all gather
 
std::shared_ptr<Work> ProcessGroupNCCL::all_gather_async(const void* sendbuff, void* recvbuff, size_t sendcount, OwnTensor::Dtype dtype, bool sync_){
    size_t type_size = OwnTensor::Tensor::dtype_size(dtype);
    const void* actual_send = static_cast<const char*>(sendbuff) + rank_ * sendcount * type_size;
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclAllGather(actual_send, recvbuff, sendcount, ncclTypeConversion(dtype), comm_, communication_stream_);
        },
        sync_
    );
}

//gather
 
std::shared_ptr<Work> ProcessGroupNCCL::gather_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclGather(sendbuff, recvbuff, count, ncclTypeConversion(dtype), root, comm_, communication_stream_);
        },
        sync_
    );
}


//scatter
 
std::shared_ptr<Work> ProcessGroupNCCL::scatter_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclScatter(sendbuff, recvbuff, count, ncclTypeConversion(dtype), root, comm_, communication_stream_);
        },
        sync_
    );

   
}

//broadcast
 
std::shared_ptr<Work> ProcessGroupNCCL::broadcast_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclBroadcast(sendbuff, recvbuff, count, ncclTypeConversion(dtype), root, comm_, communication_stream_);
        },
        sync_
    );
}


//all to all
 
std::shared_ptr<Work> ProcessGroupNCCL::alltoall_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclAlltoAll(sendbuff, recvbuff, count, ncclTypeConversion(dtype), comm_, communication_stream_);
        },
        sync_
    );
}

//point to point communications

std::shared_ptr<Work> ProcessGroupNCCL::sendrecv_async(const void* sendbuff, void* recvbuff, int send_rank, int recv_rank, size_t count, OwnTensor::Dtype dtype, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            ncclGroupStart();
            ncclResult_t nccl_send_status = ncclSend(sendbuff, count, ncclTypeConversion(dtype), recv_rank, comm_, communication_stream_);
            ncclResult_t nccl_recv_status = ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank, comm_, communication_stream_);
            // result_t send_status = send(sendbuff, count, datatype, recv_rank);
            // result_t recv_status = recieve(recvbuff, count, datatype, send_rank);
            ncclGroupEnd();
            return nccl_send_status;
        },
        sync_
    );
}


std::shared_ptr<Work> ProcessGroupNCCL::send_async(const void* sendbuff, size_t count, OwnTensor::Dtype dtype, int recv_rank, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclSend(sendbuff, count, ncclTypeConversion(dtype), recv_rank, comm_, communication_stream_);
        },
        sync_
    );
}

std::shared_ptr<Work> ProcessGroupNCCL::recieve_async(void* recvbuff, size_t count, OwnTensor::Dtype dtype, int send_rank, bool sync_){
    return launch_work_collectives(
        communication_stream_,
        [&]() -> ncclResult_t{
            return ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank, comm_, communication_stream_);
        },
        sync_
    );
}


template<typename NCCLFUNC>
std::shared_ptr<Work>
ProcessGroupNCCL::launch_work_collectives(
    cudaStream_t stream,
    NCCLFUNC nccl_op,
    bool to_sync
) {
    // Create Work object
    auto work_collectives = std::make_shared<Work>(stream, comm_);

    // Mark start
    work_collectives->begin();

    // Enqueue NCCL operation
    ncclResult_t nccl_status = nccl_op();
    work_collectives->setNcclStatus(nccl_status);

    // Record CUDA event after enqueue
    if (!work_collectives->event_record()) {
        throw std::runtime_error("Failed to record CUDA event");
    }

    // Synchronous path
    if (to_sync) {
        bool ok = work_collectives->wait();
        if (!ok) {
            throw std::runtime_error("Collective synchronization failed");
        }
        return work_collectives;
    }

    // ---------- ASYNC PATH ----------
    // Capture only safe state
    auto timeout_flag = &time_limit_exceeded;

    std::thread(
        [work_collectives, timeout_flag]() {
            const auto start =
                std::chrono::steady_clock::now();
            const auto timeout =
                std::chrono::seconds(120);

            while (true) {
                // Check progress
                if (work_collectives->query()) {
                    return;
                }

                // Timeout check
                if (std::chrono::steady_clock::now() - start > timeout) {
                    timeout_flag->store(true);
                    return;
                }

                // Small sleep to avoid CPU burn
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100));
            }
        }
    ).detach();

    return work_collectives;
}


// Explicit template instantiation (required for linking)
// The lambda types used in async operations require this

static MPI_Datatype mpiTypeConversion(OwnTensor::Dtype type) {
    switch(type){
        case OwnTensor::Dtype::Int32:   return MPI_INT;
        case OwnTensor::Dtype::Int64:   return MPI_LONG_LONG;
        case OwnTensor::Dtype::Float32: return MPI_FLOAT;
        case OwnTensor::Dtype::Float64: return MPI_DOUBLE;
        default: throw std::runtime_error("MPI Type conversion failed: unsupported type");
    }
}

static MPI_Op mpiOpConversion(op_t op) {
    switch(op){
        case sum: return MPI_SUM;
        case max: return MPI_MAX;
        case min: return MPI_MIN;
        case mul: return MPI_PROD;
        default: throw std::runtime_error("MPI Op conversion failed: unsupported op");
    }
}

void ProcessGroupNCCL::all_reduce_cpu(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, op_t operation) {
    MPI_Datatype mpi_type = mpiTypeConversion(dtype);
    MPI_Op mpi_op = mpiOpConversion(operation);
    
    int mpi_err = MPI_Allreduce(sendbuff == recvbuff ? MPI_IN_PLACE : sendbuff, 
                               recvbuff, count, mpi_type, mpi_op, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed");
    }
}
