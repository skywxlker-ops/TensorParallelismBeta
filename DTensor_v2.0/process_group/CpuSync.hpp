#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <memory>
#include <thread>




class Work{

public:

    //will only get the stream as argument. 
    explicit Work(cudaStream_t stream) : stream_(stream), completed_(true), success_(false) {
		// create event with disable timing (fast)
		cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("cudaEventCreateWithFlags failed: ") +
			cudaGetErrorString(err));
		}
	}

    ~Work() {
        // event may be destroyed after use; ignore errors in destructor
        cudaEventDestroy(event_);
    }


    /* to let the cpu know where the device is.
        mark if a work is completed.

        option 1:
            -> Use cuda Events to track the progress.
        option 2:
            -> Use Flag Polling.
    
    */

    // will create the event with a cudaEventDisableTimings flag.
    // to avoid unnecessary time recording by the cuda for events.
    bool start(){
        std::lock_guard<std::mutex> lk(mutex_);
        completed_ = false;
        success_ = false;
        return true;
    }

    // will stop the recoding. will put a asynchronous record on the kernel execution to note that the task is completed.
    // will set the completed and success as true if the execution is completed properly.
    // throws error if the status is not equals to cudaSuccess.
    inline bool stop(){
        std::unique_lock<std::mutex> lock(mutex_);
        cudaError_t err = cudaEventRecord(event_, stream_);
        if (err != cudaSuccess) {
            completed_ = true;
            success_ = false;
            return false;
        }
        completed_ = true;
        success_ = true;
        return true;
    }

    // returns if completed is true
    bool is_completed(){
        std::unique_lock<std::mutex> lock(mutex_);
        return completed_;
    }

    // will synchronize the cpu until the stream is fully executed.
    // only when the kernels in the stream is completed, the getStatus() will be called function is returned
    bool wait(){
        std::lock_guard<std::mutex> lock(mutex_);
        cudaError_t err = cudaEventSynchronize(event_);
        return (err == cudaSuccess);
    }

    //will return the success.
    bool is_success(){ 
        std::unique_lock<std::mutex> lock(mutex_);
        return success_;
    }

    bool query() const {
        // std::unique_lock<std::mutex> lk(mutex_);
        return (cudaEventQuery(event_) == cudaSuccess);
    }

    
private:
    std::mutex mutex_;
    cudaEvent_t event_;
    bool completed_ = true;
    bool success_ = false;
    cudaStream_t stream_;

 
};