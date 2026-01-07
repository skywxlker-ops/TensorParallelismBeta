/**
 * Unparalleled - Distributed Tensor Framework
 * 
 * Single umbrella header that includes all public API components.
 * 
 * Usage:
 *   #include <unparalleled/unparalleled.h>
 * 
 * Or include individual components:
 *   #include <unparalleled/dtensor.h>
 *   #include <unparalleled/process_group.h>
 */

#pragma once

// Core distributed tensor
#include "unparalleled/dtensor.h"

// Device mesh for multi-GPU topology
#include "unparalleled/device_mesh.h"

// NCCL process group for collectives
#include "unparalleled/process_group.h"

// Tensor layout and sharding
#include "unparalleled/layout.h"
