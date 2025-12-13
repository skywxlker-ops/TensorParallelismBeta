#include "./include/ProcessGroupNCCL.h"
#include <iostream>
#include <vector>
#include <memory.h>
#include <cstdint>

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
    CUDACHECK(cudaSetDevice(local_rank_));

    NCCLCHECK(ncclCommInitRank(&comm_, world_size_, id_, rank_));
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

    std::shared_ptr<Work> work_obj = std::make_shared<Work>(communication_stream);
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



//collectives

//all reduce
 
result_t ProcessGroupNCCL::all_reduce(const void* sendbuff, void* recvbuff,size_t count, OwnTensor::Dtype dtype, op_t operation){
    // auto work = std::make_shared<Work>(communication_stream_);
    while(!work_obj_->is_completed());
    work_obj_->start();
    // blockStreamEvent();
    // std::cout<<"All Reduce started: "<<std::endl;
    // work->start();
    ncclResult_t nccl_status = ncclAllReduce(sendbuff, recvbuff, count, ncclTypeConversion(dtype), ncclOperationConversion(operation), comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"All Reduce Completed."<<std::endl;
    // bool status = work_obj_->stop();
    // if(!status){
    //     return pgInternalError;
    // }
    CUDACHECK(cudaStreamSynchronize(communication_stream_));
    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            std::cout<<ncclGetErrorString(nccl_status)<<std::endl;
            return pgCudaError;
        }else if(nccl_status  == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;

    }
    
    return pgSuccess;
}

//reduce
 
result_t ProcessGroupNCCL::reduce(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, op_t op, int root){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"Reduce started: "<<std::endl;
    
    ncclResult_t nccl_status = ncclReduce(sendbuff, recvbuff, count, ncclTypeConversion(dtype), ncclOperationConversion(op), root,  comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"Reduce Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}

//reduce scatter
 
result_t ProcessGroupNCCL::reduce_scatter(const void* sendbuff, void* recvbuff, size_t recv_count, OwnTensor::Dtype dtype, op_t operation){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"Reduce Scatter started: "<<std::endl;
    ncclResult_t nccl_status = ncclReduceScatter(sendbuff, recvbuff, recv_count, ncclTypeConversion(dtype), ncclOperationConversion(operation), comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"Reduce Scatter Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}


//all gather
 
result_t ProcessGroupNCCL::all_gather(const void* sendbuff, void* recvbuff, size_t sendcount, OwnTensor::Dtype dtype){
    size_t type_size = OwnTensor::Tensor::dtype_size(dtype);
    const void* actual_send = static_cast<const char*>(sendbuff) + rank_ * sendcount * type_size;
    while(!work_obj_->is_completed());
    work_obj_->start();
    blockStreamEvent();
    // std::cout<<"All Gather started: "<<std::endl;
    ncclResult_t nccl_status = ncclAllGather(actual_send, recvbuff, sendcount, ncclTypeConversion(dtype), comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"All Gather Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}

//gather
 
result_t ProcessGroupNCCL::gather(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root){
    
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"Gather started: "<<std::endl;
    ncclResult_t nccl_status = ncclGather(sendbuff, recvbuff, count, ncclTypeConversion(dtype), root, comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"Gather Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}


//scatter
 
result_t ProcessGroupNCCL::scatter(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"Scatter started: "<<std::endl;
    ncclResult_t nccl_status = ncclScatter(sendbuff, recvbuff, count, ncclTypeConversion(dtype), root, comm_, communication_stream_);

    work_obj_->stop();
    // std::cout<<"Scatter Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}

//broadcast
 
result_t ProcessGroupNCCL::broadcast(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"BroadCast started: "<<std::endl;
    ncclResult_t nccl_status = ncclBroadcast(sendbuff, recvbuff, count, ncclTypeConversion(dtype), root, comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"BroadCast Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}


//all to all
 
result_t ProcessGroupNCCL::alltoall(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"All To All started: "<<std::endl;
    ncclResult_t nccl_status = ncclAlltoAll(sendbuff, recvbuff, count, ncclTypeConversion(dtype), comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"All To All Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}

//point to point communications

result_t ProcessGroupNCCL::sendrecv(const void* sendbuff, void* recvbuff, int send_rank, int recv_rank, size_t count, OwnTensor::Dtype dtype){
    ncclDataType_t datatype = ncclTypeConversion(dtype);
    while(!work_obj_->is_completed());
    work_obj_->start();
    ncclGroupStart();
    ncclResult_t nccl_send_status = ncclSend(sendbuff, count, ncclTypeConversion(dtype), recv_rank, comm_, communication_stream_);
    ncclResult_t nccl_recv_status = ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank, comm_, communication_stream_);
    // result_t send_status = send(sendbuff, count, datatype, recv_rank);
    // result_t recv_status = recieve(recvbuff, count, datatype, send_rank);
    ncclGroupEnd();
    work_obj_->stop();

    if(nccl_send_status != ncclSuccess || nccl_recv_status != ncclSuccess){
        if(nccl_send_status == ncclUnhandledCudaError || nccl_recv_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_send_status == ncclInternalError || nccl_recv_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}


result_t ProcessGroupNCCL::send(const void* sendbuff, size_t count, OwnTensor::Dtype dtype, int recv_rank){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"Send started: "<<std::endl;
    ncclResult_t nccl_status = ncclSend(sendbuff, count, ncclTypeConversion(dtype), recv_rank, comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"Send Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}

result_t ProcessGroupNCCL::recieve(void* recvbuff, size_t count, OwnTensor::Dtype dtype, int send_rank){
    while(!work_obj_->is_completed());
    work_obj_->start();
    // std::cout<<"Recv started: "<<std::endl;
    ncclResult_t nccl_status = ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank, comm_, communication_stream_);
    work_obj_->stop();
    // std::cout<<"Recv Completed."<<std::endl;

    if(nccl_status != ncclSuccess){
        if(nccl_status == ncclUnhandledCudaError){
            return pgCudaError;
        }else if(nccl_status == ncclInternalError){
            return pgNcclError;
        }else return pgInternalError;
    }
    return pgSuccess;
}
