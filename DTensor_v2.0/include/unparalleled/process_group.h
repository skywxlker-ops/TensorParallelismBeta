/**
 * Unparalleled - ProcessGroupNCCL
 * 
 * NCCL-based process group for collective communications.
 * Provides AllReduce, AllGather, ReduceScatter, Broadcast, etc.
 * 
 * Key function:
 *   init_process_group(world_size, rank) - Initialize NCCL communicator
 */

#pragma once

#include "process_group/ProcessGroupNCCL.h"
