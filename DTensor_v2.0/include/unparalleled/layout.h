/**
 * Unparalleled - Layout
 * 
 * Tensor layout describing sharding strategy.
 * 
 * ShardingType:
 *   REPLICATED - Tensor copied on all devices
 *   SHARDED    - Tensor split along a dimension
 *   PARTIAL    - Tensor pending reduction
 */

#pragma once

#include "tensor/layout.h"
#include "tensor/placement.h"
